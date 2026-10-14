"""Single-owner GPU execution with continuous batching and captured decode steps."""

import asyncio
import base64
import hashlib
import io
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_functional
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.codec import CodecGraph, load_codec
from drama3.model import AttentionBatch, DramaModel


@dataclass
class Request:
    request_id: str
    text: str
    reference: bytes | None
    reference_text: str
    max_frames: int
    loop: asyncio.AbstractEventLoop
    output: asyncio.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)
    slot: int = -1
    position: int = 0
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted: int = 0
    next_emit: int = 10
    last_frame: torch.Tensor | None = None
    history: torch.Tensor | None = None

    def emit(self, value: bytes | Exception | None) -> None:
        self.loop.call_soon_threadsafe(self.output.put_nowait, value)


class DecodeGraph:
    def __init__(self, model: DramaModel, size: int, first_frame: bool = False):
        self.model = model
        self.size = size
        self.first_frame = first_frame
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
        self.batch.refresh()
        if self.first_frame:
            return self.model.tower_forward(
                self.hidden,
                self.frames,
                self.positions,
                self.slots,
                self.batch,
                None,
                self.history,
            )
        return self.model.decode(self.frames, self.positions, self.slots, self.history, self.batch)

    def run(self, requests: list[Request], hidden: torch.Tensor | None = None) -> torch.Tensor:
        count = len(requests)
        self.frames[:count].copy_(torch.stack([request.last_frame for request in requests]))
        self.history[:count].copy_(torch.stack([request.history for request in requests]))
        slot_ids = [request.slot for request in requests]
        slot_ids += list(range(self.model.slots // 2 + count, self.model.slots // 2 + self.size))
        positions = [request.position for request in requests] + [0] * (self.size - count)
        self.slots.copy_(torch.tensor(slot_ids, dtype=torch.int64))
        self.positions.copy_(torch.tensor(positions, dtype=torch.int64))
        if hidden is not None:
            self.hidden.copy_(hidden)
        self.graph.replay()
        return self.output[:count].clone()


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


class Engine:
    def __init__(self, checkpoint: str, fish_source: str, slots: int = 64, max_length: int = 2048):
        if slots not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Slots must be a power of two between 1 and 64")
        self.checkpoint = Path(checkpoint)
        self.fish_source = Path(fish_source)
        self.slots = slots
        self.max_length = max_length
        self.pending = queue.Queue(maxsize=512)
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self.run, name="drama3-gpu", daemon=False)
        self.thread.start()

    @torch.inference_mode()
    def run(self) -> None:
        active = []
        try:
            torch.cuda.set_device(0)
            self.model = DramaModel(
                str(self.checkpoint), slots=2 * self.slots, max_length=self.max_length
            )
            self.codec = load_codec(self.checkpoint, self.fish_source)
            self.references = {}
            self.graphs = {}
            for size in (1, 2, 4, 8, 16, 32, 64):
                if size <= self.slots:
                    self.graphs[size] = DecodeGraph(self.model, size)
            self.first_graph = DecodeGraph(self.model, 1, first_frame=True)
            self.prefill_graphs = {
                size: PrefillGraph(self.model, size)
                for size in (128, 256, 512)
                if size <= self.max_length
            }
            self.codec_graphs = {
                (batch, length): CodecGraph(self.codec, batch, length)
                for batch in (1, 2, 4, 8)
                if batch <= self.slots
                for length in (10, 20, 32, 64, 96)
            }
            torch.cuda.synchronize()
            self.ready.set()
            free = list(range(self.slots))
            steps = 0
            while not self.stopping.is_set():
                if free and (not active or steps % 4 == 0):
                    try:
                        request = self.pending.get(timeout=0.01 if not active else 0)
                    except queue.Empty:
                        request = None
                    if request is not None:
                        if request.cancelled.is_set():
                            continue
                        request.slot = free.pop()
                        try:
                            self.prefill(request)
                            if request.max_frames == 1:
                                self.emit_audio(request)
                                request.emit(None)
                                free.append(request.slot)
                            else:
                                active.append(request)
                        except Exception as error:
                            request.emit(error)
                            free.append(request.slot)
                if not active:
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
                size = 1 << (len(active) - 1).bit_length()
                frames = self.graphs[size].run(active)
                main_tokens = frames[:, 0].cpu().tolist()
                survivors = []
                emit = []
                finished = []
                for request, frame, main in zip(active, frames, main_tokens, strict=True):
                    done = main == self.model.config.eos_token_id
                    if not done:
                        request.frames.append(frame[1:].clone())
                        request.last_frame = frame.clone()
                        request.history = torch.cat((request.history[1:], frame[:1]))
                        request.position += 1
                    done = done or len(request.frames) >= request.max_frames
                    if len(request.frames) >= request.next_emit or done:
                        emit.append(request)
                    if done:
                        finished.append(request)
                        free.append(request.slot)
                    else:
                        survivors.append(request)
                self.emit_audio_batch(emit)
                for request in finished:
                    request.emit(None)
                active = survivors
                steps += 1
        except Exception as error:
            self.failure = error
            self.ready.set()
        finally:
            for request in active:
                request.emit(self.failure or RuntimeError("Engine stopped"))
            while not self.pending.empty():
                self.pending.get_nowait().emit(self.failure or RuntimeError("Engine stopped"))

    def prefill(self, request: Request) -> None:
        codes = []
        texts = []
        if request.reference is not None:
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
        bucket = next((size for size in self.prefill_graphs if 3 <= len(frames) <= size), None)
        hidden = (
            self.prefill_graphs[bucket].run(frames, request.slot)
            if bucket is not None
            else self.model.prefill(frames, request.slot)
        )
        request.position = len(frames) - 1
        request.last_frame = frames[-1].to("cuda")
        request.history = F.pad(frames[-16:, 0], (max(0, 16 - len(frames)), 0), value=-1).to("cuda")
        first = self.first_graph.run([request], hidden)[0]
        if first[0].item() == self.model.config.eos_token_id:
            raise ValueError("The model generated no acoustic frames")
        request.frames.append(first[1:].clone())
        request.last_frame = first.clone()
        request.history = torch.cat((request.history[1:], first[:1]))
        request.position += 1

    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.codec.from_indices(codes)

    def emit_audio(self, request: Request) -> None:
        self.emit_audio_batch([request])

    def emit_audio_batch(self, requests: list[Request]) -> None:
        groups = {}
        for request in requests:
            count = len(request.frames)
            if count <= request.emitted:
                continue
            left = max(0, request.emitted - 32)
            length = count - left
            groups.setdefault(length, []).append((request, left, count))
        for length, jobs in groups.items():
            for start in range(0, len(jobs), min(8, self.slots)):
                chunk = jobs[start : start + min(8, self.slots)]
                batch = 1 << (len(chunk) - 1).bit_length()
                codes = torch.zeros((len(chunk), 10, length), dtype=torch.int64, device="cuda")
                for index, (request, left, count) in enumerate(chunk):
                    codes[index, :, : count - left] = torch.stack(request.frames[left:]).T
                graph = self.codec_graphs.get((batch, length))
                waveform = (
                    graph.run(codes) if graph is not None else self.decode_audio(codes)
                ).float()
                pcm = (waveform.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
                for index, (request, left, count) in enumerate(chunk):
                    begin, end = (request.emitted - left) * 2048, (count - left) * 2048
                    request.emit(pcm[index, 0, begin:end].astype("<i2", copy=False).tobytes())
                    request.emitted = count
                    request.next_emit = count + (
                        10 if count == 10 else 12 if count == 20 else 32 if count == 32 else 64
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
