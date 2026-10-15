"""Validate batched multi-step graphs, output limits and context boundaries."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from drama3.engine import DecodeGraph
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=32, max_length=512)
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    results = []
    for batch in (2, 4, 8):
        count = batch - int(args.partial)
        single = DecodeGraph(model, batch, requests=device)
        for steps in (2, 4):
            multiple = DecodeGraph(model, batch, steps=steps, requests=device)
            for limited in (False, True):
                snapshots = []
                for graph in (single, multiple):
                    for pair in (*model.states.values(), *model.kv.values()):
                        for tensor in pair:
                            tensor.zero_()
                    model.sampling_step.fill_(123)
                    device.codes.zero_()
                    requests = []
                    for row in range(count):
                        position = 510 if row % 2 else 30
                        frame = torch.zeros(11, dtype=torch.int64, device="cuda")
                        frame[0] = model.config.semantic_start_token_id
                        history = torch.full((16,), -1, dtype=torch.int64, device="cuda")
                        limit = 1 + row % 2 if limited else 500
                        device.initialize(row, frame, history, position, limit)
                        requests.append(SimpleNamespace(slot=row, position=position))
                    if graph is single:
                        output = torch.stack([graph.run(requests).clone() for _ in range(steps)])
                    else:
                        output = graph.run(requests).clone()
                    for pair in (*model.states.values(), *model.kv.values()):
                        for tensor in pair:
                            assert not bool(tensor[count : model.slots // 2].any()), (
                                "Unrelated primary slot changed"
                            )
                    snapshots.append(
                        dict(
                            output=output,
                            metadata=device.metadata[:batch].clone(),
                            codes=device.codes[:batch].clone(),
                            last=device.last[:batch].clone(),
                            history=device.history[:batch].clone(),
                            states=[
                                value[:batch].clone()
                                for pair in model.states.values()
                                for value in pair
                            ],
                        )
                    )
                expected, actual = snapshots
                for name in ("metadata", "codes", "last", "history"):
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
                for row in range(count):
                    generated = int(expected["metadata"][row, 1])
                    # Values after EOS/output limit are deliberately discarded;
                    # they must never become emitted codes or advance metadata.
                    compared = min(
                        steps, generated + (int(expected["metadata"][row, 3]) and not limited)
                    )
                    torch.testing.assert_close(
                        actual["output"][:compared, row],
                        expected["output"][:compared, row],
                        atol=0,
                        rtol=0,
                    )
                    if not int(expected["metadata"][row, 3]):
                        for a, e in zip(actual["states"], expected["states"], strict=True):
                            torch.testing.assert_close(a[row], e[row], atol=0, rtol=0)
                result = dict(batch=batch, count=count, steps=steps, limited=limited, passed=True)
                results.append(result)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
