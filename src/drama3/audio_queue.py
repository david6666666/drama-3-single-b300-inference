"""Bounded codec stream with event-ordered PCM delivery and input lifetimes."""

import queue
import threading
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class AudioBatch:
    codes: torch.Tensor
    rows: torch.Tensor
    pcm: torch.Tensor
    host: torch.Tensor
    done: torch.cuda.Event
    callbacks: list


class AudioQueue:
    def __init__(self, graphs, max_pending=32, threaded=False):
        self.graphs = graphs
        self.max_pending = max_pending
        self.stream = torch.cuda.Stream()
        self.threaded = threaded
        self.pending = queue.Queue(max_pending) if threaded else deque()
        self.failure = None
        self.closed = False
        if threaded:
            self.device = torch.cuda.current_device()
            self.ready = threading.Event()
            self.thread = threading.Thread(
                target=self._deliver_loop, name="drama3-pcm", daemon=False
            )
            self.thread.start()
            self.ready.wait()
            if self.failure is not None:
                raise self.failure

    def _deliver_loop(self):
        try:
            torch.cuda.set_device(self.device)
        except Exception as error:
            self.failure = error
            return
        finally:
            self.ready.set()
        while True:
            batch = self.pending.get()
            try:
                if batch is None:
                    return
                try:
                    batch.done.synchronize()
                except Exception as error:
                    self.failure = self.failure or error
                self._deliver(batch)
            except Exception as error:
                self.failure = self.failure or error
            finally:
                self.pending.task_done()

    def _deliver(self, batch):
        pcm = batch.host.numpy() if self.failure is None else None
        for index, (callback, samples) in enumerate(batch.callbacks):
            data = (
                self.failure
                if self.failure is not None
                else pcm[index, 0, :samples].astype("<i2", copy=False).tobytes()
            )
            callback(data)

    def submit(self, codes, rows, callbacks):
        if self.closed or self.failure is not None:
            raise self.failure or RuntimeError("Audio queue is closed")
        # The model stream gathers a private codes tensor before its slot can
        # be reused. Preserve it and rows until this stream's copy has finished.
        while not self.threaded and len(self.pending) >= self.max_pending:
            self.drain(block=True)
        ready = torch.cuda.Event()
        ready.record()
        self.stream.wait_event(ready)
        codes.record_stream(self.stream)
        rows.record_stream(self.stream)
        with (
            torch.cuda.stream(self.stream),
            torch.autocast("cuda", dtype=self.graphs.decoder.dtype),
        ):
            waveform = self.graphs(codes, rows=rows)
            pcm = (waveform.clamp(-1, 1) * 32767).to(torch.int16)
            host = torch.empty(pcm.shape, dtype=torch.int16, pin_memory=True)
            host.copy_(pcm, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        batch = AudioBatch(codes, rows, pcm, host, done, callbacks)
        if self.threaded:
            self.pending.put(batch)
        else:
            self.pending.append(batch)

    def drain(self, block=False):
        if self.threaded:
            if block:
                self.pending.join()
            if self.failure is not None:
                raise self.failure
            return
        if block and self.pending:
            self.pending[0].done.synchronize()
        while self.pending and self.pending[0].done.query():
            batch = self.pending.popleft()
            self._deliver(batch)

    def close(self):
        if self.closed:
            return
        if self.threaded:
            self.pending.join()
            self.pending.put(None)
            self.thread.join()
            self.closed = True
            if self.failure is not None:
                raise self.failure
            return
        while self.pending:
            self.drain(block=True)
        self.closed = True

    def abort(self, error):
        if self.threaded:
            self.failure = self.failure or error
            try:
                self.close()
            except Exception:
                pass  # The original failure is retained and delivered to consumers.
            return
        while self.pending:
            batch = self.pending.popleft()
            for callback, _ in batch.callbacks:
                callback(error)
