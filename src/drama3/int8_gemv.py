"""Diagnostic W8A16 GEMV: symmetric per-output-row INT8 weights, BF16 inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def int8_gemv_kernel(
    X, W, SCALE, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCK_K: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_K)
    x = tl.load(X + columns, columns < K, 0.0).to(tl.float32)
    weight = tl.load(
        W + rows[:, None] * K + columns[None, :], (rows[:, None] < N) & (columns[None, :] < K), 0
    ).to(tl.float32)
    value = tl.sum(weight * x[None, :], 1)
    scale = tl.load(SCALE + rows, rows < N, 0.0)
    tl.store(Y + rows, value * scale, rows < N)


def quantize_rows(weight):
    fp32 = weight.float()
    scales = fp32.abs().amax(dim=-1).clamp_min(1e-12) / 127
    quantized = (fp32 / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized, scales


def int8_gemv(x, weight, scales, rows=4, warps=4):
    n, k = weight.shape
    output = torch.empty((1, n), device=x.device, dtype=x.dtype)
    int8_gemv_kernel[(triton.cdiv(n, rows),)](
        x,
        weight,
        scales,
        output,
        n,
        k,
        rows,
        triton.next_power_of_2(k),
        num_warps=warps,
    )
    return output
