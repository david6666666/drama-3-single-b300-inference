"""Diagnostic BF16 projections with FP32 accumulation and explicit split-K.

No activation or weight quantization is introduced. The reduction writes BF16
only at the original projection boundary. This module is not a serving default.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def projection_kernel(
    X,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    mi, ni, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    k = split * tl.cdiv(K, SPLITS) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, SPLITS * BK)):
        offset = block * BK
        x = tl.load(
            X + m[:, None] * SX + k[None, :] + offset,
            (m[:, None] < M) & (k[None, :] + offset < K),
            other=0,
        )
        w = tl.load(
            W + n[None, :] * K + k[:, None] + offset,
            (n[None, :] < N) & (k[:, None] + offset < K),
            other=0,
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        P + split * M * N + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N)
    )


@triton.jit
def reduce_kernel(P, Y, ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLITS)
    value = tl.load(P + s[:, None] * ELEMENTS + i[None, :], i[None, :] < ELEMENTS, other=0)
    tl.store(Y + i, tl.sum(value, 0), i < ELEMENTS)


def projection(x, weight, bm=16, bn=64, bk=64, splits=4):
    m, k = x.shape
    n = weight.shape[0]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("This diagnostic preserves BF16 inputs and weights")
    if not weight.is_contiguous() or k % (splits * bk):
        raise ValueError("Expected contiguous weights and whole K tiles in every split")
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = (
        torch.empty((splits, m, n), device=x.device, dtype=torch.float32) if splits > 1 else output
    )
    projection_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x,
        weight,
        partial,
        m,
        n,
        k,
        x.stride(0),
        bm,
        bn,
        bk,
        splits,
        num_warps=4,
        num_stages=3,
    )
    if splits > 1:
        reduce_kernel[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output
