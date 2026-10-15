"""Validate real packed prompts, causal KV and untouched recurrent state."""

import argparse
import json
from pathlib import Path

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.engine import BatchedPrefillGraph
from drama3.model import DramaModel
from drama3.packed_prefill import PackedPrefillGraph


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
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--padded-control", action="store_true")
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=32, max_length=512)
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][:6]
    prompts = [
        build_drama3_prompt(
            s["text"],
            model.weights.tokenizer,
            model.config,
            [s["reference_text"]],
            [torch.tensor(s["reference_codes"])],
        )
        for s in samples
    ]
    results = []
    for count in (3, 5, 6):
        selected = prompts[:count]
        capacity = triton.cdiv(sum(map(len, selected)), 128) * 128
        graph = (
            BatchedPrefillGraph(model, 1 << (count - 1).bit_length(), 256)
            if args.padded_control
            else PackedPrefillGraph(model, 6, capacity)
        )
        untouched = torch.tensor(list(range(count, 25)), device="cuda")
        original_state = {
            i: (conv[untouched].clone(), state[untouched].clone())
            for i, (conv, state) in model.states.items()
        }
        original_kv = {
            i: (key[untouched].clone(), value[untouched].clone())
            for i, (key, value) in model.kv.items()
        }
        actual = graph.run(selected, list(range(count)))[:count]
        for i, (conv, state) in model.states.items():
            assert torch.equal(conv[untouched], original_state[i][0]), (count, i, "untouched conv")
            assert torch.equal(state[untouched], original_state[i][1]), (
                count,
                i,
                "untouched state",
            )
        for i, (key, value) in model.kv.items():
            assert torch.equal(key[untouched], original_kv[i][0]), (count, i, "untouched key")
            assert torch.equal(value[untouched], original_kv[i][1]), (count, i, "untouched value")
        del original_state, original_kv
        errors = []
        for row, prompt in enumerate(selected):
            reference_slot = row + 8
            expected = model.prefill(prompt, reference_slot)
            errors.append((f"hidden{row}", rrms(actual[row], expected[0])))
            for i, (conv, state) in model.states.items():
                errors.extend(
                    [
                        (f"conv{row}/{i}", rrms(conv[row], conv[reference_slot])),
                        (f"state{row}/{i}", rrms(state[row], state[reference_slot])),
                    ]
                )
            for i, (key, value) in model.kv.items():
                length = len(prompt) - (i >= 40)
                errors.extend(
                    [
                        (f"key{row}/{i}", rrms(key[row, :length], key[reference_slot, :length])),
                        (
                            f"value{row}/{i}",
                            rrms(value[row, :length], value[reference_slot, :length]),
                        ),
                    ]
                )
        passed = all(error < 0.03 for _, error in errors)
        result = dict(
            count=count, capacity=capacity, errors=errors, untouched_exact=True, passed=passed
        )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    count=count,
                    capacity=capacity,
                    worst=max(errors, key=lambda x: x[1]),
                    passed=passed,
                )
            ),
            flush=True,
        )
        assert passed, max(errors, key=lambda x: x[1])
        if count == 6:
            padded = BatchedPrefillGraph(model, 8, 256)
            # Identical prompt/slot inputs; time captured GPU work, excluding CPU copies.
            padded.run(selected, list(range(count)))
            graph.run(selected, list(range(count)))
            result["padded_graph_ms"] = triton.testing.do_bench_cudagraph(padded.graph.replay)
            result["packed_graph_ms"] = triton.testing.do_bench_cudagraph(graph.graph.replay)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "errors"}), flush=True)


if __name__ == "__main__":
    main()
