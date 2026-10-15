"""Device-resident request inputs, token history and codec frames."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load(
    last, history, metadata, slots, frames_out, history_out, positions_out, MAX_LENGTH: tl.constexpr
):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    lane = tl.arange(0, 32)
    frame = tl.load(last + slot * 11 + lane, lane < 11, 0)
    hist = tl.load(history + slot * 16 + lane, lane < 16, -1)
    pos = tl.load(metadata + slot * 4)
    tl.store(frames_out + row * 11 + lane, frame, lane < 11)
    tl.store(history_out + row * 16 + lane, hist, lane < 16)
    tl.store(positions_out + row, tl.minimum(pos, MAX_LENGTH - 1))


@triton.jit
def _save(last, history, metadata, codes, slots, output, EOS: tl.constexpr, LIMIT: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    lane = tl.arange(0, 32)
    main = tl.load(output + row * 11)
    position = tl.load(metadata + slot * 4)
    count = tl.load(metadata + slot * 4 + 1)
    limit = tl.load(metadata + slot * 4 + 2)
    done = tl.load(metadata + slot * 4 + 3)
    append = (done == 0) & (main != EOS) & (count < limit)
    frame = tl.load(output + row * 11 + lane, lane < 11, 0)
    shifted = tl.load(history + slot * 16 + lane + 1, lane < 15, -1)
    shifted = tl.where(lane == 15, main, shifted)
    acoustic = tl.load(output + row * 11 + lane + 1, lane < 10, 0)
    tl.store(last + slot * 11 + lane, frame, (lane < 11) & append)
    tl.store(history + slot * 16 + lane, shifted, (lane < 16) & append)
    tl.store(codes + (slot * LIMIT + count) * 10 + lane, acoustic, (lane < 10) & append)
    count += append.to(tl.int64)
    tl.store(metadata + slot * 4, position + append.to(tl.int64))
    tl.store(metadata + slot * 4 + 1, count)
    tl.store(metadata + slot * 4 + 3, done | (main == EOS) | (count >= limit))


@triton.jit
def _gather(
    codes, rows, output, B: tl.constexpr, T: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    book, time = index // T, index % T
    slot = tl.load(rows + row)
    length = tl.load(rows + 2 * B + row)
    start = tl.load(rows + 3 * B + row)
    value = tl.load(
        codes + (slot * LIMIT + start + time) * 10 + book, (book < 10) & (time < length), 0
    )
    tl.store(output + row * 10 * T + index, value, book < 10)


class DeviceRequests:
    def __init__(self, slots: int, eos: int, max_length: int, max_frames: int = 500):
        self.eos = eos
        self.max_frames = max_frames
        self.max_length = max_length
        self.last = torch.zeros(slots, 11, device="cuda", dtype=torch.int64)
        self.last[:, 0] = eos
        self.history = torch.full((slots, 16), -1, device="cuda", dtype=torch.int64)
        self.metadata = torch.zeros(slots, 4, device="cuda", dtype=torch.int64)
        self.metadata[:, 0] = max_length - 1
        self.metadata[:, 2] = max_frames
        self.codes = torch.zeros(slots, max_frames, 10, device="cuda", dtype=torch.int64)

    def initialize(self, slot, frame, history, position, limit):
        self.last[slot].copy_(frame)
        self.history[slot].copy_(history)
        self.metadata[slot].copy_(torch.tensor([position, 0, limit, 0]))

    def load(self, slots, frames, history, positions):
        _load[(len(slots),)](
            self.last,
            self.history,
            self.metadata,
            slots,
            frames,
            history,
            positions,
            MAX_LENGTH=self.max_length,
            num_warps=1,
        )

    def save(self, slots, frames):
        _save[(len(slots),)](
            self.last,
            self.history,
            self.metadata,
            self.codes,
            slots,
            frames,
            EOS=self.eos,
            LIMIT=self.max_frames,
            num_warps=1,
        )

    def gather_codes(self, rows, length):
        batch = rows.shape[1]
        output = torch.empty(batch, 10, length, device="cuda", dtype=torch.int64)
        _gather[(batch, triton.cdiv(10 * length, 256))](
            self.codes, rows, output, B=batch, T=length, LIMIT=self.max_frames, BLOCK=256
        )
        return output
