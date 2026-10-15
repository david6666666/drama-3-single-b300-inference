"""Bounded two-frame decode, overlapping the next backbone with the tower tail."""

import torch


def pair_blocker(active, *, waiting, pending, supported):
    """Keep admissions and scheduled audio boundaries outside a captured pair."""
    if not supported:
        return "capacity"
    if waiting or pending:
        return "admission"
    if any(min(r.next_emit, r.max_frames) - r.frame_count < 2 for r in active):
        return "audio_boundary"
    return None


class FramePipelineGraph:
    steps = 2

    def __init__(self, model, size, requests):
        from drama3.model import AttentionBatch

        if (
            model.tower.config.vq_backbone_codebooks != 1
            or len(model.tower.shared_layers) != 4
            or len(model.tower.residual_layers) != 9
        ):
            raise ValueError("Frame pipeline requires the Drama3 codebook1 backbone dependency")
        if model.moe_shared_stream is not None:
            raise ValueError("Frame pipeline has not been validated with shared-expert overlap")
        self.model, self.size, self.requests = model, size, requests
        self.slots = torch.arange(size, device="cuda", dtype=torch.int64) + model.slots // 2
        self.positions = torch.full((size,), model.max_length - 1, device="cuda", dtype=torch.int64)
        self.next_positions = self.positions.clone()
        self.frames = torch.zeros(size, 11, device="cuda", dtype=torch.int64)
        self.frames[:, 0] = model.config.eos_token_id
        self.history = torch.full((size, 16), -1, device="cuda", dtype=torch.int64)
        self.batch = AttentionBatch(self.slots, self.positions, model.max_length)
        # Tower1 still reads p while backbone2 reads/writes its own p+1 metadata.
        self.next_batch = AttentionBatch(self.slots, self.next_positions, model.max_length)
        self.capture_stream = torch.cuda.Stream()
        self.auxiliary = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self.output = self.forward()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.output = self.forward()

    def forward(self):
        model = self.model
        self.requests.load(
            self.slots,
            self.frames,
            self.history,
            self.positions,
            padding_start=model.slots // 2,
        )
        self.batch.refresh()
        hidden = model.backbone_forward(self.frames, self.positions, self.slots, self.batch)
        torch.add(self.positions, 1, out=self.next_positions)
        self.next_positions.clamp_(max=model.max_length - 1)
        self.next_batch.refresh()
        current = torch.cuda.current_stream()

        def launch_backbone(main, code0, code1):
            blank = torch.zeros_like(main)
            partial_frame = torch.stack((main, code0, code1, *(blank for _ in range(8))), dim=-1)
            # Allocated on main, consumed on auxiliary: retain it beyond this
            # callback so the graph allocator cannot reuse it for the tower tail.
            self.partial_frame = partial_frame
            self.auxiliary.wait_stream(current)
            with torch.cuda.stream(self.auxiliary):
                self.next_hidden = model.backbone_forward(
                    partial_frame, self.next_positions, self.slots, self.next_batch
                )

        frame1 = model.tower_forward(
            hidden,
            self.frames,
            self.positions,
            self.slots,
            self.batch,
            None,
            self.history,
            code1_callback=launch_backbone,
        )
        self.requests.save(self.slots, frame1)
        history2 = torch.cat((self.history[:, 1:], frame1[:, :1]), dim=-1)
        current.wait_stream(self.auxiliary)
        # Preserve steps=2 semantics: run the second frame even after first-frame
        # EOS. DeviceRequests.save masks finished rows; complete_decode stops at EOS.
        # Tower2 must increment sampling_step only after every tower1 sample ends.
        frame2 = model.tower_forward(
            self.next_hidden,
            frame1,
            self.next_positions,
            self.slots,
            self.next_batch,
            None,
            history2,
        )
        self.requests.save(self.slots, frame2)
        return torch.stack((frame1, frame2))

    def run(self, requests):
        count = len(requests)
        slot_ids = [request.slot for request in requests]
        slot_ids += list(range(self.model.slots // 2 + count, self.model.slots // 2 + self.size))
        self.slots.copy_(torch.tensor(slot_ids, dtype=torch.int64))
        self.graph.replay()
        return self.output[:, :count]
