"""Check small-batch projection math and measure captured kernels."""

import json
from functools import partial

import torch
import torch.nn.functional as F
import triton

from drama3.kernels import gemv

with torch.inference_mode():
    torch.manual_seed(123)
    for n, k in (
        (9216, 2048),
        (2048, 4096),
        (12288, 2048),
        (2048, 6144),
        (4096, 2048),
        (1024, 2048),
        (2048, 512),
        (256, 2048),
        (1, 2048),
    ):
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        expected = F.linear(x, weight)
        reference = triton.testing.do_bench_cudagraph(partial(F.linear, x, weight))
        for rows in (1, 2, 4, 8):
            actual = gemv(x, weight, rows)
            error = (
                (
                    (actual.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                )
                .sqrt()
                .item()
            )
            assert error < 0.003, error
            elapsed = triton.testing.do_bench_cudagraph(partial(gemv, x, weight, rows))
            print(
                json.dumps(
                    {
                        "n": n,
                        "k": k,
                        "rows": rows,
                        "relative_rmse": error,
                        "triton_ms": elapsed,
                        "torch_ms": reference,
                    }
                ),
                flush=True,
            )
