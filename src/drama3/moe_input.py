"""Fuse unchanged native FP8 activation preparation and router conversion."""

import torch
import triton
import triton.language as tl


@triton.jit
def prepare_kernel(X, ROUTER, Q, S, LOGITS, B: tl.constexpr, SX: tl.constexpr, SR: tl.constexpr):
    row = tl.program_id(0)
    group = tl.arange(0, 16)
    col = tl.arange(0, 128)
    x = tl.load(X + row * SX + group[:, None] * 128 + col[None, :]).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1), 1e-10) / 448.0
    q = tl.clamp(x / scale[:, None], -448.0, 448.0)
    tl.store(Q + row * 2048 + group[:, None] * 128 + col[None, :], q)
    # FlashInfer consumes the transposed scale layout. Write that layout
    # directly; the scale values and FP8 group boundaries are unchanged.
    tl.store(S + group * B + row, scale)
    expert = tl.arange(0, 256)
    logits = tl.load(ROUTER + row * SR + expert).to(tl.float32)
    tl.store(LOGITS + row * 256 + expert, logits)


def prepare_moe_input(x, router):
    if x.shape[1] != 2048 or router.shape != (len(x), 256):
        raise ValueError("Expected native Drama 3 hidden and full 256-expert router shapes")
    if x.stride(1) != 1 or router.stride(1) != 1:
        raise ValueError("Channel strides must be contiguous")
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((16, len(x)), device=x.device, dtype=torch.float32)
    logits = torch.empty(router.shape, device=x.device, dtype=torch.float32)
    prepare_kernel[(len(x),)](
        x, router, q, scales, logits, len(x), x.stride(0), router.stride(0), num_warps=4
    )
    return q, scales, logits
