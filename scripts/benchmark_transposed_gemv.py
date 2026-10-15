"""Test lossless BF16 weight transposition and split-K GEMV before integration."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

from drama3.kernels import linear


@triton.jit
def project(X, W, OUT, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    weight = tl.load(W + k[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), 0)
    value = tl.sum(weight.to(tl.float32) * x[:, None], axis=0)
    tl.store(OUT + tl.program_id(1) * N + n, value, n < N)


@triton.jit
def reduce(PART, OUT, N: tl.constexpr, S: tl.constexpr, BS: tl.constexpr):
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    s = tl.arange(0, BS)
    value = tl.load(PART + s[:, None] * N + n[None, :], (s[:, None] < S) & (n[None, :] < N), 0)
    tl.store(OUT + n, tl.sum(value, axis=0), n < N)


def transposed_gemv(x, weight, bn, bk, warps):
    k, n = weight.shape
    splits = triton.cdiv(k, bk)
    output = torch.empty((1, n), device=x.device, dtype=x.dtype)
    temporary = torch.empty((splits, n), device=x.device) if splits > 1 else output
    project[(triton.cdiv(n, bn), splits)](x, weight, temporary, n, k, bn, bk, num_warps=warps)
    if splits > 1:
        reduce[(triton.cdiv(n, 128),)](temporary, output, n, splits, triton.next_power_of_2(splits))
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for n, k in ((12352, 2048), (9216, 2048), (2048, 4096), (2048, 6144), (1408, 2048)):
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        transposed = weight.T.contiguous()
        reference = linear(x, weight)
        baseline = partial(linear, x, weight)
        hot = triton.testing.do_bench_cudagraph(baseline) * 1000
        cold = triton.testing.do_bench(baseline) * 1000
        for bn in (4, 8, 16, 32):
            for bk in (128, 512, triton.next_power_of_2(k)):
                for warps in (4, 8):
                    if bn * bk > 65536:
                        continue
                    fn = partial(transposed_gemv, x, transposed, bn, bk, warps)
                    row = dict(
                        n=n, k=k, bn=bn, bk=bk, warps=warps, baseline_hot=hot, baseline_cold=cold
                    )
                    try:
                        actual = fn()
                        rrms = float(
                            (
                                (actual.float() - reference.float()).square().mean()
                                / reference.float().square().mean()
                            ).sqrt()
                        )
                        row.update(rrms=rrms, passed=bool(actual.isfinite().all()) and rrms < 0.003)
                        if row["passed"]:
                            row.update(
                                hot_us=triton.testing.do_bench_cudagraph(fn) * 1000,
                                cold_us=triton.testing.do_bench(fn) * 1000,
                            )
                    except Exception as error:
                        row.update(passed=False, error=str(error))
                    results.append(row)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
