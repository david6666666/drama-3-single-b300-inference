"""Test shared-weight BF16 GEMV for the low-concurrency projection bottleneck.

Diagnostic only: no serving dispatch change. Reuse each weight tile for all
input rows and retain FP32 accumulation with a single BF16 output boundary.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from drama3.kernels import linear


@triton.jit
def batched_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_K)
    weight = tl.load(
        W + rows[:, None] * K + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K),
        other=0,
    ).to(tl.float32)
    for m in tl.static_range(M):
        x = tl.load(X + m * SX + columns, columns < K, other=0).to(tl.float32)
        value = tl.sum(weight * x[None, :], axis=1)
        tl.store(Y + m * N + rows, value, rows < N)


def batched_gemv(x, weight, rows, warps):
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    batched_kernel[(triton.cdiv(n, rows),)](
        x,
        weight,
        output,
        m,
        n,
        k,
        x.stride(0),
        rows,
        triton.next_power_of_2(k),
        num_warps=warps,
    )
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    results = []
    for m in (2, 4, 8):
        for n, k in (
            (12352, 2048),
            (9216, 2048),
            (2048, 4096),
            (12288, 2048),
            (2048, 6144),
            (1408, 2048),
            (2048, 512),
            (4224, 2048),
        ):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            expected = F.linear(x, w)
            baseline = triton.testing.do_bench_cudagraph(partial(linear, x, w)) * 1000
            for rows in (1, 2, 4, 8):
                for warps in (4, 8):
                    fn = partial(batched_gemv, x, w, rows, warps)
                    actual = fn()
                    rrms = float(
                        (
                            (actual.float() - expected.float()).square().mean()
                            / expected.float().square().mean()
                        ).sqrt()
                    )
                    record = dict(
                        m=m,
                        n=n,
                        k=k,
                        rows=rows,
                        warps=warps,
                        rrms=rrms,
                        passed=bool(actual.isfinite().all()) and rrms < 0.003,
                        baseline_us=baseline,
                    )
                    if record["passed"]:
                        record["microseconds"] = triton.testing.do_bench_cudagraph(fn) * 1000
                    results.append(record)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps(min(results[-8:], key=lambda r: r.get("microseconds", float("inf")))),
                flush=True,
            )


if __name__ == "__main__":
    main()
