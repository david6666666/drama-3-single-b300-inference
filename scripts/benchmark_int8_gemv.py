"""Check rowwise INT8 dense projections on checkpoint weights before any serving use."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.int8_gemv import dp4a_gemv, int8_gemv, quantize_rows
from drama3.kernels import linear
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp4a", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    choices = []
    for index in (0, 19, 39):
        layer = model.backbone.layers[index]
        if layer.layer_type == "linear_attention":
            choices.extend(
                [
                    (f"backbone{index}/input", layer.linear_attn.packed_input),
                    (f"backbone{index}/output", layer.linear_attn.out_proj.weight),
                ]
            )
        else:
            choices.extend(
                [
                    (f"backbone{index}/input", layer.self_attn.packed_qkv),
                    (f"backbone{index}/output", layer.self_attn.o_proj.weight),
                ]
            )
    tower = model.tower.residual_layers[0].transformer_layer.feed_forward
    choices.extend([("tower/up", tower.packed_gate_up), ("tower/down", tower.w2.weight)])
    results = []
    for name, weight in choices:
        n, k = weight.shape
        quantized, scales = quantize_rows(weight)
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        reference = partial(linear, x, weight)
        expected = reference()
        baseline = triton.testing.do_bench_cudagraph(reference) * 1000
        baseline_cold = triton.testing.do_bench(reference) * 1000
        for rows in (1, 2, 4, 8):
            for warps in (4, 8):
                kernel = dp4a_gemv if args.dp4a else int8_gemv
                fn = partial(kernel, x, quantized, scales, rows, warps)
                actual = fn()
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    ).sqrt()
                )
                row = dict(
                    name=name,
                    n=n,
                    k=k,
                    rows=rows,
                    warps=warps,
                    dp4a=args.dp4a,
                    rrms=error,
                    passed=bool(actual.isfinite().all()) and error < 0.015,
                    baseline_us=baseline,
                    baseline_cold_us=baseline_cold,
                )
                if row["passed"]:
                    row["int8_us"] = triton.testing.do_bench_cudagraph(fn) * 1000
                    row["int8_cold_us"] = triton.testing.do_bench(fn) * 1000
                results.append(row)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
