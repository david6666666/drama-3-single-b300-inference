"""Tune GEMV with cache flushing, since a full Drama 3 step exceeds B300 L2."""

import json
from functools import partial

import torch
import torch.nn.functional as F
import triton

from drama3.kernels import gemv_kernel


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    for n, k in (
        (9216, 2048),
        (2048, 4096),
        (12352, 2048),
        (2048, 6144),
        (4096, 2048),
        (1024, 2048),
        (2048, 512),
        (256, 2048),
    ):
        x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        expected = F.linear(x, weight)
        output = torch.empty_like(expected)
        reference = triton.testing.do_bench(partial(F.linear, x, weight))
        for rows in (1, 2, 4, 8, 16, 32):
            for warps in (4, 8):
                fn = partial(
                    gemv_kernel[(triton.cdiv(n, rows),)],
                    x,
                    weight,
                    output,
                    n,
                    k,
                    rows,
                    triton.next_power_of_2(k),
                    num_warps=warps,
                )
                fn()
                error = (
                    (
                        (output.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    )
                    .sqrt()
                    .item()
                )
                assert error < 0.003, error
                elapsed = triton.testing.do_bench(fn)
                print(
                    json.dumps(
                        {
                            "n": n,
                            "k": k,
                            "rows": rows,
                            "warps": warps,
                            "relative_rmse": error,
                            "microseconds": elapsed * 1000,
                            "torch_microseconds": reference * 1000,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
