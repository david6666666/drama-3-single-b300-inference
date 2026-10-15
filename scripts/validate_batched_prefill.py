"""Check independent sequence state under padded batched prompt processing."""

import argparse
import json
from pathlib import Path

import torch

from drama3.engine import BatchedPrefillGraph, state_pool_size
from drama3.model import DramaModel


def rrms(a, b):
    return float(
        (
            (a.float() - b.float()).square().mean() / b.float().square().mean().clamp_min(1e-12)
        ).sqrt()
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-batch", type=int, choices=(8, 16), default=8)
    parser.add_argument("--span", type=int, choices=(128, 192, 256), default=128)
    args = parser.parse_args()
    capacity = args.partial_batch
    span = args.span
    model = DramaModel(args.checkpoint, slots=state_pool_size(capacity, capacity), max_length=512)
    torch.manual_seed(42)
    lengths = torch.tensor([33, 91, 128, 3], device="cuda")
    slots = torch.arange(4, device="cuda").repeat_interleave(span)
    results = []
    for index in (0, 1, 19, 39):
        layer = model.backbone.layers[index]
        x = torch.randn(4, span, 2048, device="cuda", dtype=torch.bfloat16)
        if layer.layer_type == "linear_attention":
            conv, state = model.states[index]
            conv.zero_()
            state.zero_()
            actual = model.linear_attention(
                x.flatten(0, 1), layer.linear_attn, index, slots, False, lengths
            ).reshape(4, span, 2048)
            for row, length in enumerate(lengths.tolist()):
                conv[row + 4].zero_()
                state[row + 4].zero_()
                expected = model.linear_attention(
                    x[row, :length],
                    layer.linear_attn,
                    index,
                    torch.full((length,), row + 4, device="cuda"),
                    False,
                )
                for name, a, b in (
                    ("output", actual[row, :length], expected),
                    ("conv", conv[row], conv[row + 4]),
                    ("state", state[row], state[row + 4]),
                ):
                    result = {
                        "layer": index,
                        "row": row,
                        "check": name,
                        "relative_rmse": rrms(a, b),
                    }
                    results.append(result)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(result), flush=True)
                    assert result["relative_rmse"] < 0.03, result
    for index, attention in (
        (3, model.backbone.layers[3].self_attn),
        (40, model.tower.shared_layers[0].transformer_layer.attention),
    ):
        x = torch.randn(4, span, 2048, device="cuda", dtype=torch.bfloat16)
        actual = model.attention(
            x.flatten(0, 1),
            attention,
            index,
            torch.arange(span, device="cuda").repeat(4),
            slots,
            None,
            prefill_sequences=4,
        ).reshape(4, span, 2048)
        for row, length in enumerate(lengths.tolist()):
            expected = model.attention(
                x[row, :length],
                attention,
                index,
                torch.arange(length, device="cuda"),
                torch.full((length,), row + 4, device="cuda"),
                None,
            )
            error = rrms(actual[row, :length], expected)
            results.append(
                {"layer": index, "row": row, "check": "attention", "relative_rmse": error}
            )
            assert error < 0.03, results[-1]
    graph = BatchedPrefillGraph(model, 4, span)
    prompts = []
    for length in lengths.tolist():
        frames = torch.zeros(length, 11, dtype=torch.int64, device="cuda")
        frames[:, 0] = torch.randint(1000, 10000, (length,), device="cuda")
        prompts.append(frames)
    actual = graph.run(prompts, [0, 1, 2, 3])
    actual_states = {
        index: (conv[:4].clone(), state[:4].clone())
        for index, (conv, state) in model.states.items()
    }
    expected = model.prefill_batched(graph.frames, torch.arange(4, 8, device="cuda"), lengths)
    errors = [("hidden", rrms(actual, expected))]
    for index, (conv, state) in model.states.items():
        errors.extend(
            [
                (f"conv{index}", rrms(actual_states[index][0], conv[4:8])),
                (f"state{index}", rrms(actual_states[index][1], state[4:8])),
            ]
        )
    assert all(error < 0.03 for _, error in errors), errors
    results.append({"check": "batched_graph_vs_eager", "errors": errors, "pass": True})
    # A partially filled prefill must not touch unrelated live requests. GDN
    # creates an independent suffix sequence for each padded row, so both the
    # row and its suffix need reserved slots, including for dummy requests.
    partial_graph = BatchedPrefillGraph(model, capacity, span)
    for count in (capacity - 3, capacity - 2, capacity - 1):
        untouched = {
            index: (conv[count:capacity].clone(), state[count:capacity].clone())
            for index, (conv, state) in model.states.items()
        }
        untouched_kv = {
            index: (key[count:capacity].clone(), value[count:capacity].clone())
            for index, (key, value) in model.kv.items()
        }
        partial_graph.run([prompts[i % 4] for i in range(count)], list(range(count)))
        for index, (conv, state) in model.states.items():
            assert torch.equal(conv[count:capacity], untouched[index][0]), (count, index, "conv")
            assert torch.equal(state[count:capacity], untouched[index][1]), (count, index, "state")
        for index, (key, value) in model.kv.items():
            assert torch.equal(key[count:capacity], untouched_kv[index][0]), (count, index, "key")
            assert torch.equal(value[count:capacity], untouched_kv[index][1]), (
                count,
                index,
                "value",
            )
        results.append(
            {"check": "partial_batch_preserves_other_requests", "count": count, "pass": True}
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("Batched prefill graph validation passed", flush=True)


if __name__ == "__main__":
    main()
