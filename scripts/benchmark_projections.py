"""Compare BF16 low-M projection kernels, with output checks before timing."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from flashinfer.gemm import mm_bf16

from drama3.kernels import linear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for m in (1, 2, 4, 8, 16, 32, 64):
        for n, k in (
            (9216, 2048),
            (2048, 4096),
            (12352, 2048),
            (2048, 6144),
            (4096, 2048),
            (1024, 2048),
            (2048, 512),
            (256, 2048),
            (4097, 2048),
        ):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            expected = F.linear(x, w)
            for backend in ("current", "torch", "cute-dsl", "tgv"):
                fn = (
                    partial(linear, x, w)
                    if backend == "current"
                    else partial(F.linear, x, w)
                    if backend == "torch"
                    else partial(mm_bf16, x, w.T, backend=backend)
                )
                row = {"m": m, "n": n, "k": k, "backend": backend}
                try:
                    actual = fn()
                    error = (
                        (
                            (actual.float() - expected.float()).square().mean()
                            / expected.float().square().mean()
                        )
                        .sqrt()
                        .item()
                    )
                    row.update(relative_rmse=error, passed=error < 0.003)
                    assert row["passed"], error
                    row["microseconds"] = triton.testing.do_bench_cudagraph(fn) * 1000
                except Exception as error:
                    row.update(passed=False, error=str(error))
                print(json.dumps(row), flush=True)
                results.append(row)
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
