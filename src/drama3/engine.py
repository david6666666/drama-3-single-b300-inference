"""Single-owner GPU execution with continuous batching and captured decode steps."""

import asyncio
import base64
import hashlib
import io
import math
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_functional
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.audio_queue import AudioQueue
from drama3.codec import load_codec, streaming_codec
from drama3.model import AttentionBatch, DramaModel
from drama3.packed_prefill import PackedPrefillGraph
from drama3.profiling import StageProfiler, TimelineProfiler
from drama3.runtime_state import DeviceRequests


@dataclass
class Request:
    request_id: str
    text: str
    reference: bytes | None
    reference_text: str
    max_frames: int
    loop: asyncio.AbstractEventLoop
    output: asyncio.Queue
    queued_at: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)
    slot: int = -1
    position: int = 0
    frame_count: int = 0
    emitted: int = 0
    next_emit: int = 6
    chunk_index: int = 0
    last_frame: torch.Tensor | None = None
    history: torch.Tensor | None = None
    reference_codes: list[list[int]] | None = None
    pending_audio: int = 0
    finish_requested: bool = False
    audio_failed: bool = False
    audio_lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, value: bytes | Exception | None) -> None:
        self.loop.call_soon_threadsafe(self.output.put_nowait, value)


class DecodeGraph:
    def __init__(
        self,
        model: DramaModel,
        size: int,
        first_frame: bool = False,
        steps: int = 1,
        requests: DeviceRequests | None = None,
    ):
        self.model = model
        self.size = size
        self.first_frame = first_frame
        self.steps = steps
        self.requests = requests
        self.frames = torch.zeros((size, 11), dtype=torch.int64, device="cuda")
        self.frames[:, 0] = model.config.eos_token_id
        # Capture the largest attention plan; replay may shorten each sequence.
        self.positions = torch.full((size,), model.max_length - 1, dtype=torch.int64, device="cuda")
        self.slots = torch.arange(size, device="cuda", dtype=torch.int64) + model.slots // 2
        self.history = torch.full((size, 16), -1, dtype=torch.int64, device="cuda")
        self.hidden = torch.zeros((size, 2048), dtype=torch.bfloat16, device="cuda")
        self.batch = AttentionBatch(self.slots, self.positions, model.max_length)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.output = self.forward()

    def forward(self) -> torch.Tensor:
        if self.requests is not None:
            self.requests.load(self.slots, self.frames, self.history, self.positions)
        self.batch.refresh()
        if self.first_frame:
            output = self.model.tower_forward(
                self.hidden,
                self.frames,
                self.positions,
                self.slots,
                self.batch,
                None,
                self.history,
            )
            if self.requests is not None:
                self.requests.save(self.slots, output)
            return output
        if self.steps == 1:
            output = self.model.decode(
                self.frames, self.positions, self.slots, self.history, self.batch
            )
            if self.requests is not None:
                self.requests.save(self.slots, output)
            return output
        frames, history = self.frames, self.history
        outputs = []
        for _ in range(self.steps):
            self.batch.refresh()
            frames = self.model.decode(frames, self.positions, self.slots, history, self.batch)
            if self.requests is not None:
                self.requests.save(self.slots, frames)
            outputs.append(frames)
            history = torch.cat((history[:, 1:], frames[:, :1]), dim=-1)
            self.positions.add_(1).clamp_(max=self.model.max_length - 1)
        return torch.stack(outputs)

    def run(self, requests: list[Request], hidden: torch.Tensor | None = None) -> torch.Tensor:
        count = len(requests)
        if self.requests is None:
            self.frames[:count].copy_(torch.stack([request.last_frame for request in requests]))
            self.history[:count].copy_(torch.stack([request.history for request in requests]))
        slot_ids = [request.slot for request in requests]
        slot_ids += list(range(self.model.slots // 2 + count, self.model.slots // 2 + self.size))
        positions = [request.position for request in requests] + [0] * (self.size - count)
        self.slots.copy_(torch.tensor(slot_ids, dtype=torch.int64))
        if self.requests is None:
            self.positions.copy_(torch.tensor(positions, dtype=torch.int64))
        if hidden is not None:
            self.hidden.copy_(hidden)
        self.graph.replay()
        output = self.output[:count] if self.steps == 1 else self.output[:, :count]
        return output if self.requests is not None else output.clone()


class PrefillGraph:
    """Pad prompts while masking recurrent updates after their true end."""

    def __init__(self, model: DramaModel, size: int):
        self.size = size
        self.frames = torch.zeros((size, 11), dtype=torch.int64, device="cuda")
        self.frames[:, 0] = model.config.eos_token_id
        self.positions = torch.arange(size, device="cuda")
        self.slots = torch.full((size,), model.slots - 1, dtype=torch.int64, device="cuda")
        self.length = torch.tensor([size], dtype=torch.int64, device="cuda")
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = model.prefill_padded(
                    self.frames, self.positions, self.slots, self.length
                )
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.output = model.prefill_padded(self.frames, self.positions, self.slots, self.length)

    def run(self, frames: torch.Tensor, slot: int) -> torch.Tensor:
        if not 3 <= len(frames) <= self.size:
            raise ValueError("Prefill graph requires 3 <= prompt length <= bucket size")
        self.frames[: len(frames)].copy_(frames)
        self.slots.fill_(slot)
        self.length.fill_(len(frames))
        self.graph.replay()
        return self.output.clone()


class BatchedPrefillGraph:
    def __init__(self, model, batch, span):
        self.model, self.batch, self.span = model, batch, span
        self.frames = torch.zeros(batch, span, 11, device="cuda", dtype=torch.int64)
        self.frames[:, :, 0] = model.config.eos_token_id
        self.slots = torch.arange(model.slots - batch, model.slots, device="cuda")
        self.lengths = torch.full((batch,), span, device="cuda", dtype=torch.int64)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = model.prefill_batched(self.frames, self.slots, self.lengths)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.output = model.prefill_batched(self.frames, self.slots, self.lengths)

    def run(self, prompts, slots):
        count = len(prompts)
        if count > self.batch or count != len(slots):
            raise ValueError("Prefill batch capacity exceeded or slot count differs")
        lengths = [len(prompt) for prompt in prompts]
        if min(lengths) < 3 or max(lengths) > self.span:
            raise ValueError("Prompt does not fit the captured prefill bucket")
        for index, prompt in enumerate(prompts):
            self.frames[index, : len(prompt)].copy_(prompt)
        padding = self.batch - count
        # Padding needs its own primary slot AND its own suffix scratch slot.
        # An upper-half primary would map its suffix back into active requests.
        padding_end = self.model.slots // 2
        padding_slots = list(range(padding_end - padding, padding_end))
        if set(slots).intersection(padding_slots):
            raise ValueError("Prefill padding requires reserved primary state slots")
        self.slots.copy_(torch.tensor(slots + padding_slots))
        self.lengths.copy_(torch.tensor(lengths + [3] * padding))
        self.graph.replay()
        return self.output.clone()


def state_pool_size(request_slots, prefill_limit=8):
    """Each live or padding primary slot owns a distinct GDN suffix slot."""
    return 2 * (request_slots + min(request_slots, prefill_limit))


class Engine:
    def __init__(self, checkpoint: str, fish_source: str, slots: int = 64, max_length: int = 2048):
        if slots not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Slots must be a power of two between 1 and 64")
        self.checkpoint = Path(checkpoint)
        self.fish_source = Path(fish_source)
        self.slots = slots
        self.max_length = max_length
        self.codec_precision = os.environ.get("DRAMA3_CODEC_DTYPE", "bfloat16")
        if self.codec_precision not in ("float16", "bfloat16"):
            raise ValueError("DRAMA3_CODEC_DTYPE must be float16 or bfloat16")
        self.prefill_max_batch = int(os.environ.get("DRAMA3_PREFILL_MAX_BATCH", "8"))
        self.packed_prefill = os.environ.get("DRAMA3_PACKED_PREFILL") == "1"
        self.batched_multistep = os.environ.get("DRAMA3_BATCHED_MULTISTEP") == "1"
        self.mixed_backbone = os.environ.get("DRAMA3_MIXED_BACKBONE") == "1"
        self.mixed_span256 = os.environ.get("DRAMA3_MIXED_SPAN256") == "1"
        if self.mixed_backbone and (not self.packed_prefill or self.prefill_max_batch != 8):
            raise ValueError("Mixed backbone currently requires packed prefill and batch limit 8")
        if self.mixed_backbone and os.environ.get("DRAMA3_PREFILL_MOE") != "flashinfer":
            raise ValueError("Mixed backbone requires explicit DRAMA3_PREFILL_MOE=flashinfer")
        if self.packed_prefill and self.max_length < 512:
            raise ValueError("Packed prefill currently requires a context of at least 512")
        if self.packed_prefill and os.environ.get("DRAMA3_PREFILL_MOE_TUNED") == "1":
            raise ValueError("Packed prefill currently requires DRAMA3_PREFILL_MOE_TUNED=0")
        self.prefill_batch_sizes = (
            (1, 2, 4, 6, 8, 16)
            if os.environ.get("DRAMA3_PREFILL_BATCH_6") == "1"
            else (1, 2, 4, 8, 16)
        )
        self.prefill_spans = (
            (128, 192, 256, 512)
            if os.environ.get("DRAMA3_PREFILL_BUCKET_192") == "1"
            else (128, 256, 512)
        )
        if self.prefill_max_batch not in (8, 16):
            raise ValueError("Prefill maximum batch must be 8 or 16")
        self.admission_min_batch = int(os.environ.get("DRAMA3_ADMISSION_MIN_BATCH", "1"))
        self.admission_min_running = int(os.environ.get("DRAMA3_ADMISSION_MIN_RUNNING", "32"))
        self.admission_max_wait = (
            float(os.environ.get("DRAMA3_ADMISSION_MAX_WAIT_MS", "100")) / 1000
        )
        if (
            not 1 <= self.admission_min_batch <= self.prefill_max_batch
            or self.admission_min_running < 1
            or not math.isfinite(self.admission_max_wait)
            or self.admission_max_wait < 0
        ):
            raise ValueError("Invalid admission batch, active threshold or maximum wait")
        self.pending = queue.Queue(maxsize=512)
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.failure = None
        profile_path = os.environ.get("DRAMA3_PROFILE_PATH")
        timeline_path = os.environ.get("DRAMA3_TIMELINE_PATH")
        if profile_path and timeline_path:
            raise ValueError("Select either synchronized stage profiling or event timeline")
        self.async_codec = os.environ.get("DRAMA3_ASYNC_CODEC") == "1"
        self.audio_delivery_thread = os.environ.get("DRAMA3_AUDIO_DELIVERY_THREAD") == "1"
        if self.audio_delivery_thread and not self.async_codec:
            raise ValueError("A PCM delivery thread requires DRAMA3_ASYNC_CODEC=1")
        self.audio_queue = None
        if self.async_codec and profile_path:
            raise ValueError("StageProfiler is synchronous; use a CUDA trace for codec overlap")
        self.profiler = (
            TimelineProfiler(timeline_path)
            if timeline_path
            else StageProfiler(profile_path)
            if profile_path
            else None
        )
        self.thread = threading.Thread(target=self.run, name="drama3-gpu", daemon=False)
        self.thread.start()

    @torch.inference_mode()
    def run(self) -> None:
        active = []
        waiting = []
        try:
            torch.cuda.set_device(0)
            self.model = DramaModel(
                str(self.checkpoint),
                # Reserve a full prefill batch in the primary half. The second
                # half mirrors every primary slot for padded GDN suffixes.
                slots=state_pool_size(self.slots, self.prefill_max_batch),
                max_length=self.max_length,
            )
            self.codec = load_codec(self.checkpoint, self.fish_source)
            self.device_requests = DeviceRequests(
                self.model.slots, self.model.config.eos_token_id, self.max_length
            )
            self.references = {}
            self.graphs = {}
            for size in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64):
                if size <= self.slots:
                    self.graphs[size] = DecodeGraph(self.model, size, requests=self.device_requests)
            self.first_graphs = {
                size: DecodeGraph(self.model, size, first_frame=True, requests=self.device_requests)
                for size in self.prefill_batch_sizes
                if size <= min(self.slots, self.prefill_max_batch)
            }
            self.serial_graphs = {
                steps: DecodeGraph(self.model, 1, steps=steps, requests=self.device_requests)
                for steps in (2, 4)
            }
            self.multistep_graphs = {
                (size, steps): DecodeGraph(
                    self.model, size, steps=steps, requests=self.device_requests
                )
                for size in (2, 4, 8)
                if self.batched_multistep and size <= self.slots
                for steps in (2, 4)
            }
            self.prefill_graphs = {
                size: PrefillGraph(self.model, size)
                for size in self.prefill_spans
                if size <= self.max_length
            }
            self.batched_prefill_graphs = {
                (batch, span): BatchedPrefillGraph(self.model, batch, span)
                for batch in self.prefill_batch_sizes
                if not self.packed_prefill and 1 < batch <= min(self.slots, self.prefill_max_batch)
                for span in self.prefill_spans
                if span <= self.max_length and (batch <= 8 or span <= 256)
            }
            self.packed_prefill_graphs = {}
            if self.packed_prefill and self.slots > 1:
                batch = min(self.slots, self.prefill_max_batch)
                # Replay is serial and run() clones its output before another
                # graph can reuse these temporary allocations.
                pool = torch.cuda.graph_pool_handle()
                for capacity in (128, *range(256, batch * 512 + 1, 256)):
                    self.packed_prefill_graphs[capacity] = PackedPrefillGraph(
                        self.model, batch, capacity, pool=pool
                    )
            self.mixed_graphs = {}
            self.mixed_tower_graphs = {}
            if self.mixed_backbone:
                from drama3.mixed_graph import MixedGraph

                pool = torch.cuda.graph_pool_handle()
                for size in (16, 32, 48, 64):
                    if size > self.slots:
                        continue
                    self.mixed_tower_graphs[size] = DecodeGraph(
                        self.model, size, first_frame=True, requests=self.device_requests
                    )
                    # Larger prompts/capacities use the established serial path.
                    for capacity in range(256, 2049, 256):
                        for span in (256, 512) if self.mixed_span256 else (512,):
                            self.mixed_graphs[(size, capacity, span)] = MixedGraph(
                                self.model,
                                self.device_requests,
                                size,
                                8,
                                capacity,
                                pool=pool,
                                max_length=span,
                            )
            self.codec_graphs = streaming_codec(
                self.codec, self.slots, dtype=getattr(torch, self.codec_precision)
            )
            if self.async_codec:
                self.audio_queue = AudioQueue(
                    self.codec_graphs,
                    threaded=self.audio_delivery_thread,
                    profiler=self.profiler if isinstance(self.profiler, TimelineProfiler) else None,
                )
            torch.cuda.synchronize()
            self.ready.set()
            free = list(range(self.slots))
            while not self.stopping.is_set():
                if self.audio_queue is not None:
                    self.audio_queue.drain(block=not active)
                mixed_hidden = None
                if self.mixed_backbone:
                    live = []
                    for request in active:
                        if request.cancelled.is_set():
                            free.append(request.slot)
                        else:
                            live.append(request)
                    active = live
                if free:
                    while len(waiting) < min(self.prefill_max_batch, len(free)):
                        try:
                            request = self.pending.get(
                                timeout=0.01 if not active and not waiting else 0
                            )
                        except queue.Empty:
                            break
                        if request.cancelled.is_set():
                            continue
                        waiting.append(request)
                    admit = waiting and (
                        len(active) < self.admission_min_running
                        or len(waiting) >= self.admission_min_batch
                        or time.monotonic() - waiting[0].queued_at >= self.admission_max_wait
                    )
                    arriving, prompts = [], []
                    if admit:
                        for request in waiting:
                            if request.cancelled.is_set():
                                continue
                            request.slot = free.pop()
                            try:
                                prompt = self.prepare_prompt(request)
                                arriving.append(request)
                                prompts.append(prompt)
                            except Exception as error:
                                request.emit(error)
                                free.append(request.slot)
                        waiting = []
                    if arriving:
                        try:
                            total = sum(map(len, prompts))
                            mixed_size = next(
                                (size for size in self.mixed_tower_graphs if size >= len(active)),
                                None,
                            )
                            capacity = ((total + 255) // 256) * 256
                            span = (
                                256
                                if self.mixed_span256
                                and max([*map(len, prompts), capacity - total]) <= 256
                                else 512
                            )
                            mixed_graph = self.mixed_graphs.get((mixed_size, capacity, span))
                            if (
                                len(active) > 8
                                and mixed_graph is not None
                                and all(3 <= len(prompt) <= 512 for prompt in prompts)
                            ):
                                with self.measure("mixed_backbone", len(active)):
                                    mixed_hidden, hidden = mixed_graph.run(
                                        active, prompts, [r.slot for r in arriving]
                                    )
                                batch = next(
                                    size for size in self.first_graphs if size >= len(arriving)
                                )
                                with self.measure("first_frame", len(arriving)):
                                    self.finish_prefill(arriving, hidden[:batch], batch)
                                # Advance old requests once, then admit the new ones.
                                with self.measure("mixed_decode_tower", len(active)):
                                    frames = self.mixed_tower_graphs[mixed_size].run(
                                        active, mixed_hidden
                                    )
                                active = self.complete_decode(active, frames, free)
                            else:
                                with self.measure("prefill", len(arriving)):
                                    self.prefill_many(arriving, prompts)
                        except Exception as error:
                            for request in arriving:
                                request.emit(error)
                            raise
                        for request in arriving:
                            if request.frame_count == 0:
                                request.emit(ValueError("The model generated no acoustic frames"))
                                free.append(request.slot)
                                continue
                            if request.max_frames == 1:
                                self.emit_audio(request)
                                self.finish_audio(request)
                                free.append(request.slot)
                            else:
                                active.append(request)
                if mixed_hidden is not None or not active:
                    continue
                live = []
                for request in active:
                    if request.cancelled.is_set():
                        free.append(request.slot)
                    else:
                        live.append(request)
                active = live
                if not active:
                    continue
                size = next(size for size in self.graphs if size >= len(active))
                graph = self.graphs[size]
                if len(active) == 1 and self.pending.empty():
                    request = active[0]
                    available = min(request.next_emit, request.max_frames) - request.frame_count
                    steps = 4 if available >= 4 else 2 if available >= 2 else 1
                    graph = self.serial_graphs.get(steps, graph)
                elif self.batched_multistep and size <= 8 and not waiting and self.pending.empty():
                    available = min(
                        min(request.next_emit, request.max_frames) - request.frame_count
                        for request in active
                    )
                    steps = 4 if available >= 4 else 2 if available >= 2 else 1
                    graph = self.multistep_graphs.get((size, steps), graph)
                with self.measure("decode", len(active)):
                    frames = graph.run(active)
                active = self.complete_decode(active, frames, free)
        except Exception as error:
            self.failure = error
            self.ready.set()
        finally:
            if self.audio_queue is not None:
                try:
                    if self.failure is None:
                        self.audio_queue.close()
                    else:
                        self.audio_queue.abort(self.failure)
                except Exception as error:
                    self.failure = self.failure or error
                    self.audio_queue.abort(self.failure)
            if self.profiler is not None:
                self.profiler.save()
            for request in active:
                request.emit(self.failure or RuntimeError("Engine stopped"))
            for request in waiting:
                request.emit(self.failure or RuntimeError("Engine stopped"))
            while not self.pending.empty():
                self.pending.get_nowait().emit(self.failure or RuntimeError("Engine stopped"))

    def complete_decode(self, active, frames, free):
        if frames.ndim == 2:
            frames = frames.unsqueeze(0)
        main_tokens = frames[:, :, 0].T.cpu().tolist()
        emit = []
        finished = []
        survivors = []
        for index, request in enumerate(active):
            done = False
            for main in main_tokens[index]:
                done = main == self.model.config.eos_token_id
                if not done:
                    request.frame_count += 1
                    request.position += 1
                done = done or request.frame_count >= request.max_frames
                if done:
                    break
            if request.frame_count >= request.next_emit or done:
                emit.append(request)
            if done:
                finished.append(request)
                free.append(request.slot)
            else:
                survivors.append(request)
        with self.measure("codec_submit" if self.async_codec else "codec", len(emit)):
            self.emit_audio_batch(emit)
        for request in finished:
            self.finish_audio(request)
        return survivors

    def measure(self, stage, batch):
        return (
            self.profiler.measure(stage, batch)
            if self.profiler is not None and batch
            else nullcontext()
        )

    def prepare_prompt(self, request: Request) -> torch.Tensor:
        codes = []
        texts = []
        if request.reference_codes is not None:
            codes = [torch.tensor(request.reference_codes, dtype=torch.int64)]
            texts = [request.reference_text]
        elif request.reference is not None:
            digest = hashlib.sha256(request.reference).hexdigest()
            if digest not in self.references:
                waveform, sample_rate = sf.read(
                    io.BytesIO(request.reference), dtype="float32", always_2d=True
                )
                if len(waveform) / sample_rate > 60:
                    raise ValueError("Reference audio exceeds 60 seconds")
                waveform = torch.from_numpy(waveform.mean(1)).to("cuda").view(1, 1, -1)
                if sample_rate != 44100:
                    waveform = audio_functional.resample(waveform, sample_rate, 44100)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded, _ = self.codec.encode(
                        waveform, torch.tensor([waveform.shape[-1]], device="cuda")
                    )
                if len(self.references) >= 256:
                    self.references.pop(next(iter(self.references)))
                self.references[digest] = encoded[0].cpu()
            codes = [self.references[digest]]
            texts = [request.reference_text]
        frames = build_drama3_prompt(
            request.text, self.model.weights.tokenizer, self.model.config, texts, codes
        )
        if len(frames) + request.max_frames > self.max_length:
            raise ValueError("Prompt plus output limit exceeds context capacity")
        request.position = len(frames) - 1
        request.last_frame = frames[-1].to("cuda")
        request.history = F.pad(frames[-16:, 0], (max(0, 16 - len(frames)), 0), value=-1).to("cuda")
        self.device_requests.initialize(
            request.slot, request.last_frame, request.history, request.position, request.max_frames
        )
        return frames

    def prefill_many(self, requests, prompts):
        if self.packed_prefill_graphs and len(requests) > 1:
            packed = [(r, p) for r, p in zip(requests, prompts, strict=True) if 3 <= len(p) <= 512]
            if len(packed) > 1:
                packed_requests, packed_prompts = map(list, zip(*packed, strict=True))
                total = sum(map(len, packed_prompts))
                capacity = next(size for size in self.packed_prefill_graphs if size >= total)
                hidden = self.packed_prefill_graphs[capacity].run(
                    packed_prompts, [r.slot for r in packed_requests]
                )
                batch = next(size for size in self.first_graphs if size >= len(packed))
                self.finish_prefill(packed_requests, hidden[:batch], batch)
                remaining = [
                    (r, p) for r, p in zip(requests, prompts, strict=True) if not 3 <= len(p) <= 512
                ]
                if not remaining:
                    return
                requests, prompts = map(list, zip(*remaining, strict=True))
        groups = {}
        for request, frames in zip(requests, prompts, strict=True):
            bucket = next((size for size in self.prefill_graphs if 3 <= len(frames) <= size), None)
            # Long prompts use the eager single-sequence fallback.
            key = bucket if bucket is not None else (None, request.slot)
            groups.setdefault(key, []).append((request, frames))
        chunks = []
        for key, jobs in groups.items():
            limit = self.prefill_max_batch if isinstance(key, int) and key <= 256 else 8
            chunks.extend(
                (key, jobs[start : start + limit]) for start in range(0, len(jobs), limit)
            )
        for key, jobs in chunks:
            requests, frames = map(list, zip(*jobs, strict=True))
            batch = next(size for size in self.first_graphs if size >= len(jobs))
            if batch > 1:
                hidden = self.batched_prefill_graphs[(batch, key)].run(
                    frames, [request.slot for request in requests]
                )
            elif isinstance(key, int):
                hidden = self.prefill_graphs[key].run(frames[0], requests[0].slot)
            else:
                hidden = self.model.prefill(frames[0], requests[0].slot)
            self.finish_prefill(requests, hidden, batch)

    def finish_prefill(self, requests, hidden, batch):
        first = self.first_graphs[batch].run(requests, hidden)[:, 0].cpu().tolist()
        for request, main in zip(requests, first, strict=True):
            if main != self.model.config.eos_token_id:
                request.frame_count = 1
                request.position += 1

    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.codec.from_indices(codes)

    def emit_audio(self, request: Request) -> None:
        self.emit_audio_batch([request])

    def finish_audio(self, request):
        with request.audio_lock:
            request.finish_requested = True
            if (
                not request.pending_audio
                and not request.audio_failed
                and not request.cancelled.is_set()
            ):
                request.emit(None)

    @staticmethod
    def audio_ready(request, data):
        with request.audio_lock:
            request.pending_audio -= 1
            request.audio_failed = request.audio_failed or isinstance(data, Exception)
            if not request.cancelled.is_set():
                request.emit(data)
                if (
                    request.finish_requested
                    and not request.pending_audio
                    and not request.audio_failed
                ):
                    request.emit(None)

    def emit_audio_batch(self, requests: list[Request]) -> None:
        groups = {}
        for request in requests:
            count = request.frame_count
            if count <= request.emitted:
                continue
            length = count - request.emitted
            bucket = self.codec_graphs.frame_bucket(length)
            groups.setdefault(bucket, []).append((request, request.emitted, count))
        for length, jobs in groups.items():
            start = 0
            for batch in self.codec_graphs.split(length, len(jobs)):
                chunk = jobs[start : start + batch]
                start += batch
                rows = torch.tensor(
                    [
                        [request.slot for request, _, _ in chunk],
                        [int(left == 0) for _, left, _ in chunk],
                        [count - left for _, left, count in chunk],
                        [left for _, left, _ in chunk],
                    ],
                    dtype=torch.int64,
                    device="cuda",
                )
                codes = self.device_requests.gather_codes(rows, length)
                if self.audio_queue is not None:
                    callbacks = []
                    for request, left, count in chunk:
                        with request.audio_lock:
                            request.pending_audio += 1
                        callbacks.append(
                            (partial(self.audio_ready, request), (count - left) * 2048)
                        )
                    self.audio_queue.submit(codes, rows, callbacks)
                else:
                    with torch.autocast("cuda", dtype=self.codec_graphs.decoder.dtype):
                        waveform = self.codec_graphs(codes, rows=rows)
                    pcm = (waveform.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
                for index, (request, left, count) in enumerate(chunk):
                    if self.audio_queue is None:
                        end = (count - left) * 2048
                        request.emit(pcm[index, 0, :end].astype("<i2", copy=False).tobytes())
                    request.emitted = count
                    request.chunk_index += 1
                    ramp = (6, 8, 12, 16, 24, 32)
                    request.next_emit = count + (
                        ramp[request.chunk_index] if request.chunk_index < len(ramp) else 64
                    )

    def submit(self, request: Request) -> None:
        if not self.ready.is_set() or self.failure is not None or self.stopping.is_set():
            raise RuntimeError(f"Engine unavailable: {self.failure}")
        self.pending.put_nowait(request)

    def close(self) -> None:
        self.stopping.set()
        self.thread.join()


def decode_reference(value: str) -> bytes:
    if not value.startswith("data:audio/") or ";base64," not in value:
        raise ValueError("Reference audio must be an inline audio data URL")
    return base64.b64decode(value.split(";base64,", 1)[1], validate=True)
