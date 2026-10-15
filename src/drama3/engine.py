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

from drama3.codec import load_codec, streaming_codec
from drama3.model import AttentionBatch, DramaModel
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
            self.device_requests = DeviceRequests(
                self.model.slots, self.model.config.eos_token_id, self.max_length
            )
            self.references = {}
            self.graphs = {}
            for size in (1, 2, 4, 8, 16, 32, 64):
                if size <= self.slots:
                    self.graphs[size] = DecodeGraph(self.model, size, requests=self.device_requests)
            self.first_graph = DecodeGraph(
                self.model, 1, first_frame=True, requests=self.device_requests
            )
            self.serial_graphs = {
                steps: DecodeGraph(self.model, 1, steps=steps, requests=self.device_requests)
                for steps in (2, 4)
            }
            self.prefill_graphs = {
                size: PrefillGraph(self.model, size)
                for size in (128, 256, 512)
                if size <= self.max_length
            }
            self.codec_graphs = streaming_codec(self.codec, self.slots)
            torch.cuda.synchronize()
            self.ready.set()
            free = list(range(self.slots))
            while not self.stopping.is_set():
                if free:
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
                graph = self.graphs[size]
                if len(active) == 1 and self.pending.empty():
                    request = active[0]
                    available = min(request.next_emit, request.max_frames) - request.frame_count
                    steps = 4 if available >= 4 else 2 if available >= 2 else 1
                    graph = self.serial_graphs.get(steps, graph)
                frames = graph.run(active)
                if frames.ndim == 2:
                    frames = frames.unsqueeze(0)
                main_tokens = frames[:, :, 0].T.cpu().tolist()
                survivors = []
                emit = []
                finished = []
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
                self.emit_audio_batch(emit)
                for request in finished:
                    request.emit(None)
                active = survivors
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
        bucket = next((size for size in self.prefill_graphs if 3 <= len(frames) <= size), None)
        hidden = (
            self.prefill_graphs[bucket].run(frames, request.slot)
            if bucket is not None
            else self.model.prefill(frames, request.slot)
        )
        request.position = len(frames) - 1
        request.last_frame = frames[-1].to("cuda")
        request.history = F.pad(frames[-16:, 0], (max(0, 16 - len(frames)), 0), value=-1).to("cuda")
        self.device_requests.initialize(
            request.slot, request.last_frame, request.history, request.position, request.max_frames
        )
        first = self.first_graph.run([request], hidden)[0]
        if first[0].item() == self.model.config.eos_token_id:
            raise ValueError("The model generated no acoustic frames")
        request.frame_count = 1
        request.position += 1

    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.codec.from_indices(codes)

    def emit_audio(self, request: Request) -> None:
        self.emit_audio_batch([request])

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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    waveform = self.codec_graphs(codes, rows=rows)
                pcm = (waveform.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
                for index, (request, left, count) in enumerate(chunk):
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
