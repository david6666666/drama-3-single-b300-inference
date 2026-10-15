"""Exercise EOS, output limits, slot reuse and codec gathers in device request storage."""

import json

import torch

from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    state = DeviceRequests(8, eos=999, max_length=512, max_frames=8)
    slots = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int64)
    histories = torch.arange(48, device="cuda").view(3, 16)
    for row, slot in enumerate(slots.tolist()):
        state.initialize(slot, torch.arange(11, device="cuda"), histories[row], 30 + row, row + 2)
    expected = [[], [], []]
    last = torch.empty(3, 11, device="cuda", dtype=torch.int64)
    history = torch.empty(3, 16, device="cuda", dtype=torch.int64)
    positions = torch.empty(3, device="cuda", dtype=torch.int64)
    for step in range(6):
        output = torch.arange(33, device="cuda").view(3, 11) + 100 * (step + 1)
        output[1, 0] = 999 if step >= 1 else 123
        state.save(slots, output)
        for row in range(3):
            if len(expected[row]) < row + 2 and not (row == 1 and step >= 1):
                expected[row].append(output[row].clone())
        state.load(slots, last, history, positions)
        for row, values in enumerate(expected):
            assert torch.equal(last[row], values[-1])
            assert positions[row] == 30 + row + len(values)
            target = torch.cat((histories[row], torch.stack(values)[:, 0]))[-16:]
            assert torch.equal(history[row], target)
    rows = torch.tensor([[5, 1, 3], [1, 1, 1], [2, 1, 4], [0, 0, 0]], device="cuda")
    codes = state.gather_codes(rows, 8)
    for row, values in enumerate(expected):
        assert torch.equal(codes[row, :, : len(values)], torch.stack(values)[:, 1:].T)
        assert torch.count_nonzero(codes[row, :, len(values) :]) == 0
    state.initialize(5, last[0], histories[0], 50, 1)
    fresh = torch.arange(11, device="cuda")[None] + 500
    state.save(slots[:1], fresh)
    assert torch.equal(state.codes[5, 0], fresh[0, 1:])
    assert state.metadata[5].tolist() == [51, 1, 1, 1]
    print(
        json.dumps(
            {"checks": ["EOS", "limit", "history", "gather_padding", "slot_reuse"], "pass": True}
        )
    )


if __name__ == "__main__":
    main()
