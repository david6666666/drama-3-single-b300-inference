"""Experimental FP32 expert reduction without an eight-way output temporary.

Adapted from Fish Speech fp8_moe.py at 541566932abbaebcb0a40e562a1f846738383f38.
Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC.
The activation/weight quantization and blockwise FP32 accumulation are retained;
the down projection atomically adds routed values into each token's FP32 row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def down_kernel(
    X,
    XS,
    W,
    WS,
    ROUTING,
    IDS,
    EXPERTS,
    TOTAL,
    OUT,
    PAIRS: tl.constexpr,
    M: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m_tiles = tl.cdiv(tl.load(TOTAL), M)
    if pid >= m_tiles * 16:
        return
    group = pid // (GROUP_M * 16)
    first = group * GROUP_M
    size = tl.minimum(m_tiles - first, GROUP_M)
    mi = first + (pid % (GROUP_M * 16)) % size
    ni = (pid % (GROUP_M * 16)) // size
    pairs = tl.load(IDS + mi * M + tl.arange(0, M))
    expert = tl.load(EXPERTS + mi)
    columns = ni * 128 + tl.arange(0, 128)
    inner = tl.arange(0, 128)
    acc = tl.zeros((M, 128), dtype=tl.float32)
    for block in range(4):
        x = tl.load(
            X + pairs[:, None] * 512 + block * 128 + inner[None, :], pairs[:, None] < PAIRS, 0.0
        )
        w = tl.load(W + expert * 2048 * 512 + columns[None, :] * 512 + block * 128 + inner[:, None])
        xs = tl.load(XS + pairs * 4 + block, pairs < PAIRS, 0)
        ws = tl.load(WS + expert * 64 + columns // 128 * 4 + block)
        acc += tl.dot(x, w) * xs[:, None] * ws[None, :]
    routing = tl.load(ROUTING + pairs, pairs < PAIRS, 0).to(tl.float32)
    tl.atomic_add(
        OUT + (pairs[:, None] // 8) * 2048 + columns[None, :],
        acc * routing[:, None],
        pairs[:, None] < PAIRS,
        sem="relaxed",
    )


def atomic_down(x, scales, experts, routing, ids, expert_ids, total, tokens, block_m, group_m):
    output = torch.zeros(tokens, 2048, device=x.device, dtype=torch.float32)
    down_kernel[(triton.cdiv(ids.numel(), block_m) * 16,)](
        x,
        scales,
        experts.w2_weight,
        experts.w2_scale_inv,
        routing,
        ids,
        expert_ids,
        total,
        output,
        tokens * 8,
        block_m,
        group_m,
        num_warps=4,
        num_stages=3,
    )
    return output.to(torch.bfloat16)
