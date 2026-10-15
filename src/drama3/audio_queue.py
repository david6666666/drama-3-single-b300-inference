"""Bounded codec stream with event-ordered PCM delivery and input lifetimes."""

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
    def __init__(self, graphs, max_pending=32):
        self.graphs = graphs
        self.max_pending = max_pending
        self.stream = torch.cuda.Stream()
        self.pending = deque()

    def submit(self, codes, rows, callbacks):
        # The model stream gathers a private codes tensor before its slot can
        # be reused. Preserve it and rows until this stream's copy has finished.
        while len(self.pending) >= self.max_pending:
            self.drain(block=True)
        ready = torch.cuda.Event()
        ready.record()
        self.stream.wait_event(ready)
        codes.record_stream(self.stream)
        rows.record_stream(self.stream)
        with torch.cuda.stream(self.stream), torch.autocast("cuda", dtype=torch.bfloat16):
            waveform = self.graphs(codes, rows=rows)
            pcm = (waveform.clamp(-1, 1) * 32767).to(torch.int16)
            host = torch.empty(pcm.shape, dtype=torch.int16, pin_memory=True)
            host.copy_(pcm, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        self.pending.append(AudioBatch(codes, rows, pcm, host, done, callbacks))

    def drain(self, block=False):
        if block and self.pending:
            self.pending[0].done.synchronize()
        while self.pending and self.pending[0].done.query():
            batch = self.pending.popleft()
            pcm = batch.host.numpy()
            for index, (callback, samples) in enumerate(batch.callbacks):
                callback(pcm[index, 0, :samples].astype("<i2", copy=False).tobytes())

    def close(self):
        while self.pending:
            self.drain(block=True)

    def abort(self, error):
        while self.pending:
            batch = self.pending.popleft()
            for callback, _ in batch.callbacks:
                callback(error)
