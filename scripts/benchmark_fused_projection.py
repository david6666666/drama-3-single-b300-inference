"""Check BF16 activation boundaries before timing fused single-request gates."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.fused_projection import gated_gemv
from drama3.kernels import linear, silu_mul


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for k, mode in ((512, "swiglu"), (6144, "swiglu"), (4096, "sigmoid")):
        n = 2048
        packed = torch.randn(1, 2 * k, device="cuda", dtype=torch.bfloat16)
        gate, x = packed.chunk(2, dim=-1)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5

        def reference(packed=packed, mode=mode, x=x, gate=gate, weight=weight):
            activated = silu_mul(packed) if mode == "swiglu" else x * gate.sigmoid()
            return linear(activated, weight)

        expected = reference()
        baseline = triton.testing.do_bench_cudagraph(reference) * 1000
        for rows in (1, 2, 4, 8):
            for warps in (4, 8):
                fn = partial(gated_gemv, x, gate, weight, mode, rows, warps)
                actual = fn()
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    ).sqrt()
                )
                row = dict(
                    k=k,
                    mode=mode,
                    rows=rows,
                    warps=warps,
                    rrms=error,
                    passed=bool(actual.isfinite().all()) and error < 0.001,
                    baseline_us=baseline,
                )
                if row["passed"]:
                    row["fused_us"] = triton.testing.do_bench_cudagraph(fn) * 1000
                results.append(row)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
