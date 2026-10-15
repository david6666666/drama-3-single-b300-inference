"""Diagnostic W8A16 GEMV: symmetric per-output-row INT8 weights, BF16 inputs."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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


@triton.jit
def quantize_input_kernel(X, Q, SCALE, K: tl.constexpr):
    group = tl.program_id(0)
    columns = group * 128 + tl.arange(0, 128)
    x = tl.load(X + columns, columns < K, 0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 127.0
    quantized = libdevice.nearbyint(x / scale)
    tl.store(Q + columns, tl.minimum(tl.maximum(quantized, -127), 127), columns < K)
    tl.store(SCALE + group, scale)


@triton.jit
def dp4a_gemv_kernel(
    X, XS, W, WS, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCK_K: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_K // 4)
    x = tl.load(X + columns, columns < K // 4, 0)
    w = tl.load(
        W + rows[:, None] * (K // 4) + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K // 4),
        0,
    )
    products = tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[w, x[None, :], 0],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    groups = tl.arange(0, BLOCK_K // 128)
    partial = tl.sum(products.reshape(ROWS, BLOCK_K // 128, 32), 2).to(tl.float32)
    xs = tl.load(XS + groups, groups < K // 128, 0.0)
    scale = tl.load(WS + rows, rows < N, 0.0)
    result = tl.sum(partial * xs[None, :], 1) * scale
    tl.store(Y + rows, result, rows < N)


def dp4a_gemv(x, weight, scales, rows=4, warps=4):
    n, k = weight.shape
    if k % 128:
        raise ValueError("DP4A GEMV needs complete 128-element quantization groups")
    quantized = torch.empty_like(x, dtype=torch.int8)
    input_scales = torch.empty(k // 128, device=x.device, dtype=torch.float32)
    quantize_input_kernel[(k // 128,)](x, quantized, input_scales, k, num_warps=4)
    output = torch.empty((1, n), device=x.device, dtype=x.dtype)
    dp4a_gemv_kernel[(triton.cdiv(n, rows),)](
        quantized.view(torch.int32),
        input_scales,
        weight.view(torch.int32),
        scales,
        output,
        n,
        k,
        rows,
        triton.next_power_of_2(k),
        num_warps=warps,
    )
    return output


@triton.jit
def tensor_gemv_kernel(X, W, SCALE, Y, N: tl.constexpr, K: tl.constexpr, TILE_N: tl.constexpr):
    columns = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
    inner = tl.arange(0, 128)
    acc = tl.zeros((16, TILE_N), dtype=tl.float32)
    for block in range(K // 128):
        x = tl.load(X + block * 128 + inner)
        a = tl.broadcast_to(x[None, :], (16, 128))
        b = tl.load(
            W + columns[None, :] * K + block * 128 + inner[:, None], columns[None, :] < N, 0
        ).to(X.dtype.element_ty)
        acc = tl.dot(a, b, acc)
    first = tl.arange(0, 16) == 0
    value = tl.sum(tl.where(first[:, None], acc, 0.0), 0)
    scale = tl.load(SCALE + columns, columns < N, 0.0)
    tl.store(Y + columns, value * scale, columns < N)


def tensor_gemv(x, weight, scales, rows=64, warps=4):
    n, k = weight.shape
    if k % 128:
        raise ValueError("Tensor GEMV needs complete 128-element groups")
    output = torch.empty((1, n), device=x.device, dtype=x.dtype)
    tensor_gemv_kernel[(triton.cdiv(n, rows),)](
        x,
        weight,
        scales,
        output,
        n,
        k,
        rows,
        num_warps=warps,
        num_stages=3,
    )
    return output
