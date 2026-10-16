"""Packed prompt metadata and convolution, with private padding state slots."""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class PackedLayout:
    boundaries: torch.Tensor
    sequence_slots: torch.Tensor
    max_length: int
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    def initialize_chunks(self, lengths):
        if os.environ.get("DRAMA3_COMPACT_GDN") != "1":
            return
        count = len(lengths)
        capacity = min(
            count * triton.cdiv(self.max_length, 64), triton.cdiv(sum(lengths), 64) + count - 1
        )
        self.chunk_indices = torch.empty(
            (capacity, 2), dtype=torch.int64, device=self.boundaries.device
        )
        self.chunk_offsets = torch.empty(
            count + 1, dtype=torch.int64, device=self.boundaries.device
        )
        self.update_chunks(lengths)

    def update_chunks(self, lengths):
        if self.chunk_indices is None:
            return
        indices, offsets = [], [0]
        for sequence, length in enumerate(lengths):
            indices.extend((sequence, chunk) for chunk in range(triton.cdiv(length, 64)))
            offsets.append(len(indices))
        if len(indices) > len(self.chunk_indices):
            raise ValueError("Packed chunks exceed captured descriptor capacity")
        # The remaining entries address an out-of-range chunk of a valid
        # sequence; guarded kernels skip it without touching state or output.
        invalid = (len(lengths) - 1, triton.cdiv(lengths[-1], 64))
        indices.extend([invalid] * (len(self.chunk_indices) - len(indices)))
        self.chunk_indices.copy_(torch.tensor(indices, dtype=torch.int64))
        self.chunk_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))


@triton.jit
def row_metadata(
    CU,
    SLOTS,
    POS,
    ROW_SLOTS,
    LABELS,
    FRAMES,
    EOS: tl.constexpr,
    SEQUENCES: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sequence = tl.full((BLOCK,), 0, tl.int32)
    for i in tl.static_range(1, SEQUENCES):
        sequence += (row >= tl.load(CU + i)).to(tl.int32)
    start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
    slot = tl.load(SLOTS + sequence)
    tl.store(POS + row, row - start, row < CAPACITY)
    tl.store(ROW_SLOTS + row, slot, row < CAPACITY)
    for code in tl.static_range(11):
        label = tl.load(
            FRAMES + (row + 1) * 11 + code,
            (row + 1 < end) & (row < CAPACITY),
            other=EOS if code == 0 else 0,
        )
        tl.store(LABELS + row * 11 + code, label, row < CAPACITY)


@triton.jit
def convolution(
    X,
    W,
    CU,
    SLOTS,
    STATE,
    OUT,
    STRIDE: tl.constexpr,
    TIME: tl.constexpr,
    CHANNELS: tl.constexpr,
    GUARD_EMPTY: tl.constexpr = False,
):
    sequence = tl.program_id(2)
    start, end = tl.load(CU + sequence), tl.load(CU + sequence + 1)
    if GUARD_EMPTY and tl.program_id(0) * TIME >= end - start:
        return
    t = tl.program_id(0) * TIME + tl.arange(0, TIME)
    c = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
    length = end - start
    value = tl.full((TIME, CHANNELS), 0, tl.float32)
    for tap in tl.static_range(4):
        source = t - 3 + tap
        x = tl.load(
            X + (start + source[:, None]) * STRIDE + c[None, :],
            (source[:, None] >= 0) & (source[:, None] < length),
            other=0,
        ).to(tl.float32)
        weight = tl.load(W + c * 4 + tap).to(tl.float32)
        value += x * weight[None, :]
    value = value.to(X.dtype.element_ty).to(tl.float32)
    tl.store(
        OUT + (start + t[:, None]) * 8192 + c[None, :],
        value * tl.sigmoid(value),
        t[:, None] < length,
    )
    history = t - length + 3
    slot = tl.load(SLOTS + sequence)
    original = tl.load(X + (start + t[:, None]) * STRIDE + c[None, :], t[:, None] < length, other=0)
    tl.store(
        STATE + (slot * 8192 + c[None, :]) * 3 + history[:, None],
        original,
        (history[:, None] >= 0) & (history[:, None] < 3) & (t[:, None] < length),
    )


def packed_conv(x, weight, state, layout):
    output = torch.empty((len(x), 8192), device=x.device, dtype=x.dtype)
    convolution[(triton.cdiv(layout.max_length, 8), 64, len(layout.sequence_slots))](
        x,
        weight,
        layout.boundaries,
        layout.sequence_slots,
        state,
        output,
        x.stride(0),
        8,
        128,
        GUARD_EMPTY=os.environ.get("DRAMA3_PACKED_CONV_GUARD") == "1",
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


class PackedPrefillGraph:
    def __init__(self, model, batch, capacity, max_length=512, pool=None, capture_stream=None):
        self.model, self.batch, self.capacity = model, batch, capacity
        self.frames = torch.zeros(capacity, 11, device="cuda", dtype=torch.int64)
        self.frames[:, 0] = model.config.eos_token_id
        self.labels = torch.empty_like(self.frames)
        self.positions = torch.empty(capacity, device="cuda", dtype=torch.int64)
        self.row_slots = torch.empty_like(self.positions)
        sizes = [capacity // (batch + 1)] * batch
        sizes.append(capacity - sum(sizes))
        if max(sizes) > max_length:
            raise ValueError("Capture layout exceeds maximum sequence length")
        boundaries = torch.tensor(
            [0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda", dtype=torch.int32
        )
        slots = torch.arange(model.slots - batch - 1, model.slots, device="cuda")
        self.layout = PackedLayout(boundaries, slots, max_length)
        self.layout.initialize_chunks(sizes)
        self.graph = torch.cuda.CUDAGraph()
        stream = capture_stream or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph, pool=pool, stream=capture_stream):
            self.output = self.forward()

    def forward(self):
        row_metadata[(triton.cdiv(self.capacity, 128),)](
            self.layout.boundaries,
            self.layout.sequence_slots,
            self.positions,
            self.row_slots,
            self.labels,
            self.frames,
            self.model.config.eos_token_id,
            self.batch + 1,
            self.capacity,
            128,
        )
        return self.model.prefill_packed(
            self.frames, self.labels, self.positions, self.row_slots, self.layout
        )

    def run(self, prompts, slots):
        if not prompts or len(prompts) != len(slots) or len(prompts) > self.batch:
            raise ValueError("Invalid packed prefill batch")
        lengths = list(map(len, prompts))
        padding = self.capacity - sum(lengths)
        if min(lengths) < 3 or max(lengths + [padding]) > self.layout.max_length or padding < 0:
            raise ValueError("Prompts exceed packed prefill capacity")
        if len(set(slots)) != len(slots) or min(slots) < 0 or max(slots) >= self.model.slots // 2:
            raise ValueError("Packed prefill requires distinct primary slots")
        # One private slot per missing request and one for the trailing padding.
        reserved = list(range(self.model.slots - (self.batch + 1 - len(slots)), self.model.slots))
        lengths += [0] * (self.batch - len(lengths)) + [padding]
        boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        self.layout.boundaries.copy_(torch.tensor(boundaries, dtype=torch.int32))
        self.layout.sequence_slots.copy_(torch.tensor(slots + reserved))
        self.layout.update_chunks(lengths)
        for index, prompt in enumerate(prompts):
            self.frames[boundaries[index] : boundaries[index + 1]].copy_(prompt)
        self.graph.replay()
        return self.output.clone()
