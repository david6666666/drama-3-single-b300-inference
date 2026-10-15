"""Isolate packed operators using identical real model inputs and native references.

Full-network single/batch differences are recorded separately: routing makes
them unsuitable for attributing an error to a newly introduced operator.
This diagnostic does not replace the full-corpus WER/SIM acceptance gate.
"""

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

    def check(name, actual, expected, exact=False):
        error = rrms(actual, expected)
        passed = bool(torch.isfinite(actual).all()) and (
            torch.equal(actual, expected) if exact else error < 0.03
        )
        row = dict(name=name, rrms=error, exact=exact, passed=passed)
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        assert passed, row

    def run_case(count):
        selected = prompts[:count]
        lengths = list(map(len, selected))
        capacity = triton.cdiv(sum(lengths), 128) * 128
        graph = PackedPrefillGraph(model, 6, capacity)
        actual = graph.run(selected, list(range(count)))[:count]
        check(
            f"b{count}/positions",
            graph.positions[: sum(lengths)],
            torch.cat([torch.arange(n, device="cuda") for n in lengths]),
            exact=True,
        )
        check(
            f"b{count}/slots",
            graph.row_slots[: sum(lengths)],
            torch.cat([torch.full((n,), i, device="cuda") for i, n in enumerate(lengths)]),
            exact=True,
        )
        offset = 0
        for row, prompt in enumerate(selected):
            check(
                f"b{count}/labels{row}",
                graph.labels[offset : offset + len(prompt) - 1],
                prompt[1:].cuda(),
                exact=True,
            )
            offset += len(prompt)
        owned = set(range(count)) | set(range(32 - (7 - count), 32))
        untouched = sorted(set(range(32)) - owned)
        state_before = {
            i: (c[:count].clone(), s[:count].clone(), c[untouched].clone(), s[untouched].clone())
            for i, (c, s) in model.states.items()
        }
        kv_before = {
            i: (k[untouched].clone(), v[untouched].clone()) for i, (k, v) in model.kv.items()
        }
        captured = []
        original_linear, original_attention, original_moe = (
            model.linear_attention,
            model.attention,
            model.moe,
        )
        layer_indices = {id(layer.mlp): i for i, layer in enumerate(model.backbone.layers)}

        def capture_linear(x, layer, index, slots, decode, valid_length=None):
            saved = x.clone()
            out = original_linear(x, layer, index, slots, decode, valid_length)
            conv, state = model.states[index]
            captured.append(
                (
                    "gdn",
                    index,
                    layer,
                    saved,
                    out.clone(),
                    conv[:count].clone(),
                    state[:count].clone(),
                )
            )
            return out

        def capture_attention(x, layer, index, positions, slots, batch, prefill_sequences=1):
            saved = x.clone()
            out = original_attention(x, layer, index, positions, slots, batch, prefill_sequences)
            captured.append(("attention", index, layer, saved, out.clone()))
            return out

        def capture_moe(x, layer, decode=False):
            saved = x.clone()
            out = original_moe(x, layer, decode)
            captured.append(("moe", layer_indices[id(layer)], layer, saved, out.clone()))
            return out

        model.linear_attention, model.attention, model.moe = (
            capture_linear,
            capture_attention,
            capture_moe,
        )
        try:
            expected = graph.forward()[:count]
        finally:
            model.linear_attention, model.attention, model.moe = (
                original_linear,
                original_attention,
                original_moe,
            )
        check(f"b{count}/graph_vs_eager", actual, expected)
        for index, (conv, state) in model.states.items():
            c, s, untouched_c, untouched_s = state_before[index]
            check(f"b{count}/graph_conv/{index}", conv[:count], c)
            check(f"b{count}/graph_state/{index}", state[:count], s)
            check(f"b{count}/untouched_conv/{index}", conv[untouched], untouched_c, exact=True)
            check(f"b{count}/untouched_state/{index}", state[untouched], untouched_s, exact=True)
        for index, (key, value) in model.kv.items():
            check(
                f"b{count}/untouched_key/{index}", key[untouched], kv_before[index][0], exact=True
            )
            check(
                f"b{count}/untouched_value/{index}",
                value[untouched],
                kv_before[index][1],
                exact=True,
            )
        del state_before, kv_before
        for kind, index, layer, x, output, *saved_state in captured:
            start = 0
            for row, length in enumerate(lengths):
                reference_slot = row + 8
                slots = torch.full((length,), reference_slot, device="cuda", dtype=torch.int64)
                sample = x[start : start + length].contiguous()
                if kind == "gdn":
                    conv, state = model.states[index]
                    conv[reference_slot].zero_()
                    state[reference_slot].zero_()
                    expected = original_linear(sample, layer, index, slots, False)
                    check(
                        f"b{count}/gdn_conv/{index}/{row}",
                        saved_state[0][row],
                        conv[reference_slot],
                    )
                    check(
                        f"b{count}/gdn_state/{index}/{row}",
                        saved_state[1][row],
                        state[reference_slot],
                    )
                elif kind == "attention":
                    expected = original_attention(
                        sample, layer, index, torch.arange(length, device="cuda"), slots, None
                    )
                else:
                    expected = original_moe(sample, layer, False)
                check(f"b{count}/{kind}/{index}/{row}", output[start : start + length], expected)
                start += length
        captured.clear()
        padded = BatchedPrefillGraph(model, 1 << (count - 1).bit_length(), 256)
        padded.run(selected, list(range(count)))
        graph.run(selected, list(range(count)))
        row = dict(
            count=count,
            capacity=capacity,
            padded_ms=triton.testing.do_bench_cudagraph(padded.graph.replay),
            packed_ms=triton.testing.do_bench_cudagraph(graph.graph.replay),
            passed=True,
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    for count in (3, 5, 6):
        run_case(count)


if __name__ == "__main__":
    main()
