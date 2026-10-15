"""Opt-in lossy dense projections. Separate from the default BF16 runtime."""

import torch
import triton
import triton.language as tl

WEIGHTS = {}


@triton.jit
def _quantize(W, Q, S, N: tl.constexpr, K: tl.constexpr):
    row, column = tl.program_id(0), tl.program_id(1)
    n = row * 128 + tl.arange(0, 128)
    k = column * 128 + tl.arange(0, 128)
    w = tl.load(W + n[:, None] * K + k[None, :], n[:, None] < N, 0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.max(tl.abs(w), 0), 0), 1e-8) / 448.0
    tl.store(Q + n[:, None] * K + k[None, :], w / scale)
    tl.store(S + row * (K // 128) + column, scale)


@triton.jit
def _gemv(X, W, S, OUT, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    x = tl.load(X + k).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :], n[:, None] < N, 0.0).to(tl.float32)
    scale = tl.load(S + (n[:, None] // 128) * (K // 128) + k[None, :] // 128, n[:, None] < N, 0)
    y = tl.sum(w * scale * x[None, :], 1)
    tl.store(OUT + n, y, n < N)


def register(weight):
    n, k = weight.shape
    if k % 128:
        raise ValueError("Experimental dense FP8 needs a 128-aligned reduction width")
    padded = triton.cdiv(n, 128) * 128
    quantized = torch.empty(padded, k, device=weight.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded // 128, k // 128, device=weight.device, dtype=torch.float32)
    _quantize[(padded // 128, k // 128)](weight, quantized, scales, n, k, num_warps=8)
    WEIGHTS[weight.data_ptr()] = (weight, quantized, scales)


def linear(x, weight):
    _, quantized, scales = WEIGHTS[weight.data_ptr()]
    if len(x) == 1:
        n, k = weight.shape
        rows = 2 if k >= 4096 else 4
        output = torch.empty(1, n, device=x.device, dtype=x.dtype)
        _gemv[(triton.cdiv(n, rows),)](x, quantized, scales, output, n, k, rows, num_warps=4)
        return output
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import quant_kernel, quantize_groups
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    original = len(x)
    if original % 4:
        x = torch.nn.functional.pad(x, (0, 0, 0, 4 - original % 4))
    activation, activation_scales = quantize_groups(quant_kernel, x, x.shape[-1])
    output = gemm_fp8_nt_groupwise(
        activation,
        quantized,
        activation_scales.T.contiguous().T,
        scales,
        scale_major_mode="K",
        backend="cutlass",
    )
    return output[:original, : weight.shape[0]]
