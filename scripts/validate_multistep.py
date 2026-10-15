"""Compare multi-frame CUDA graph replay with sequential one-frame execution."""

import argparse
import json
from pathlib import Path

import torch

from drama3.engine import DecodeGraph, Request
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=4, max_length=512)
    single = DecodeGraph(model, 1)
    device_requests = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    result = []
    for steps in (2, 4):
        multiple = DecodeGraph(model, 1, steps=steps)
        resident = DecodeGraph(model, 1, steps=steps, requests=device_requests)
        for position in (30, 510):
            outputs = []
            states = []
            for graph in (single, multiple, resident):
                for conv, state in model.states.values():
                    conv.zero_()
                    state.zero_()
                for keys, values in model.kv.values():
                    keys.zero_()
                    values.zero_()
                model.sampling_step.fill_(123)
                request = Request("test", "text", None, "", 500, None, None)
                request.slot = 0
                request.position = position
                request.last_frame = torch.zeros(11, dtype=torch.int64, device="cuda")
                request.last_frame[0] = model.config.semantic_start_token_id
                request.history = torch.full((16,), -1, dtype=torch.int64, device="cuda")
                if graph is resident:
                    device_requests.initialize(
                        0, request.last_frame, request.history, position, request.max_frames
                    )
                if graph is single:
                    frames = []
                    for _ in range(steps):
                        frame = graph.run([request])[0]
                        frames.append(frame)
                        request.last_frame = frame
                        request.history = torch.cat((request.history[1:], frame[:1]))
                        request.position = min(request.position + 1, 511)
                    output = torch.stack(frames)
                else:
                    output = graph.run([request])[:, 0]
                outputs.append(output.clone())
                states.append(
                    [value[0].clone() for pair in model.states.values() for value in pair]
                )
            for index in (1, 2):
                torch.testing.assert_close(outputs[0], outputs[index], atol=0, rtol=0)
                for expected, actual in zip(states[0], states[index], strict=True):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            row = {"steps": steps, "position": position, "passed": True}
            result.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
