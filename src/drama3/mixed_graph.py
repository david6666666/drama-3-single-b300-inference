"""Captured mixed backbone with disjoint decode, prompt and padding state."""

import torch
import triton

from drama3.mixed_model import mixed_backbone
from drama3.model import AttentionBatch
from drama3.packed_prefill import PackedLayout, row_metadata


class MixedGraph:
    def __init__(
        self,
        model,
        requests,
        decode_size,
        batch,
        capacity,
        pool=None,
        joint_backbone=True,
        max_length=512,
    ):
        if not 3 <= max_length <= 512:
            raise ValueError("Mixed prompt span must be between 3 and 512")
        self.model, self.requests = model, requests
        self.joint_backbone = joint_backbone
        self.decode_size, self.batch, self.capacity = decode_size, batch, capacity
        self.frames = torch.zeros(decode_size + capacity, 11, device="cuda", dtype=torch.int64)
        self.frames[:, 0] = model.config.eos_token_id
        self.positions = torch.zeros(decode_size + capacity, device="cuda", dtype=torch.int64)
        self.positions[:decode_size] = model.max_length - 1
        self.slots = torch.empty_like(self.positions)
        self.slots[:decode_size] = torch.arange(decode_size, device="cuda") + model.slots // 2
        self.history = torch.empty(decode_size, 16, device="cuda", dtype=torch.int64)
        self.labels = torch.empty(capacity, 11, device="cuda", dtype=torch.int64)
        sizes = [capacity // (batch + 1)] * batch
        sizes.append(capacity - sum(sizes))
        if max(sizes) > max_length:
            raise ValueError("Capture layout exceeds the mixed prompt span")
        self.layout = PackedLayout(
            torch.tensor(
                [0, *torch.tensor(sizes).cumsum(0).tolist()], device="cuda", dtype=torch.int32
            ),
            torch.tensor([*range(batch), model.slots - 1], device="cuda", dtype=torch.int64),
            max_length,
        )
        self.layout.initialize_chunks(sizes)
        self.attention = AttentionBatch(
            self.slots[:decode_size], self.positions[:decode_size], model.max_length
        )
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph, pool=pool):
            self.output = self.forward()

    def forward(self):
        d = self.decode_size
        self.requests.load(self.slots[:d], self.frames[:d], self.history, self.positions[:d])
        self.attention.refresh()
        row_metadata[(triton.cdiv(self.capacity, 128),)](
            self.layout.boundaries,
            self.layout.sequence_slots,
            self.positions[d:],
            self.slots[d:],
            self.labels,
            self.frames[d:],
            self.model.config.eos_token_id,
            self.batch + 1,
            self.capacity,
            128,
        )
        for conv, state in self.model.states.values():
            conv.index_fill_(0, self.layout.sequence_slots, 0)
            state.index_fill_(0, self.layout.sequence_slots, 0)
        if self.joint_backbone:
            hidden = mixed_backbone(
                self.model, self.frames, self.positions, self.slots, self.attention, self.layout, d
            )
        else:
            prefill = self.model.backbone_forward(
                self.frames[d:], self.positions[d:], self.slots[d:], None, self.layout
            )
            decode = self.model.backbone_forward(
                self.frames[:d], self.positions[:d], self.slots[:d], self.attention
            )
            hidden = torch.cat((decode, prefill))
        # Keep the existing teacher-forced prompt tower and sampling order.
        self.model.tower_forward(
            hidden[d:],
            self.frames[d:],
            self.positions[d:],
            self.slots[d:],
            None,
            self.labels,
            prefill_sequences=self.layout,
        )
        ends = self.layout.boundaries[1:-1].to(torch.int64) - 1
        return hidden[:d].clone(), hidden[d:].index_select(0, ends.clamp_min(0))

    def prepare(self, active, prompts, slots):
        count, d = len(active), self.decode_size
        if not 1 <= count <= d or not 1 <= len(prompts) <= self.batch or len(prompts) != len(slots):
            raise ValueError("Invalid mixed request count")
        lengths = list(map(len, prompts))
        padding = self.capacity - sum(lengths)
        if min(lengths) < 3 or max(lengths + [padding]) > self.layout.max_length or padding < 0:
            raise ValueError("Prompts exceed mixed graph capacity")
        decode_slots = [r.slot for r in active]
        decode_slots += list(range(self.model.slots // 2 + count, self.model.slots // 2 + d))
        reserved = list(range(self.model.slots - (self.batch + 1 - len(slots)), self.model.slots))
        pref_slots = slots + reserved
        if (
            len(set(decode_slots + pref_slots)) != len(decode_slots + pref_slots)
            or min(slots) < 0
            or max(slots) >= self.model.slots // 2
        ):
            raise ValueError("Mixed decode, prompt and padding state slots must be disjoint")
        lengths += [0] * (self.batch - len(lengths)) + [padding]
        boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        self.slots[:d].copy_(torch.tensor(decode_slots, dtype=torch.int64))
        self.layout.boundaries.copy_(torch.tensor(boundaries, dtype=torch.int32))
        self.layout.sequence_slots.copy_(torch.tensor(pref_slots, dtype=torch.int64))
        self.layout.update_chunks(lengths)
        for i, prompt in enumerate(prompts):
            self.frames[d + boundaries[i] : d + boundaries[i + 1]].copy_(prompt)

    def run(self, active, prompts, slots):
        self.prepare(active, prompts, slots)
        self.graph.replay()
        # Shared graph-pool storage cannot escape to a subsequent replay.
        return tuple(value.clone() for value in self.output)
