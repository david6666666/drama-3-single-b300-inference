"""Causal four-tap GDN prefill convolution directly on packed projection rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_conv(
    X,
    W,
    SLOTS,
    LENGTHS,
    STATE,
    OUT,
    SPAN: tl.constexpr,
    STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    TIME: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    sequence = tl.program_id(2)
    t = tl.program_id(0) * TIME + tl.arange(0, TIME)
    c = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
    value = tl.full((TIME, CHANNELS), 0, tl.float32)
    for tap in tl.static_range(4):
        source_t = t - 3 + tap
        x = tl.load(
            X + (sequence * SPAN + source_t[:, None]) * STRIDE + c[None, :],
            (source_t[:, None] >= 0) & (source_t[:, None] < SPAN),
            0.0,
        ).to(tl.float32)
        w = tl.load(W + c * 4 + tap).to(tl.float32)
        value += x * w[None, :]
    value = value.to(X.dtype.element_ty).to(tl.float32)
    tl.store(
        OUT + (sequence * SPAN + t[:, None]) * 8192 + c[None, :],
        value * tl.sigmoid(value),
        t[:, None] < SPAN,
    )
    slot = tl.load(SLOTS + sequence * SLOT_STRIDE)
    length = tl.load(LENGTHS + sequence) if HAS_LENGTHS else SPAN
    history = t - length + 3
    original = tl.load(
        X + (sequence * SPAN + t[:, None]) * STRIDE + c[None, :], t[:, None] < SPAN, 0.0
    )
    tl.store(
        STATE + (slot * 8192 + c[None, :]) * 3 + history[:, None],
        original,
        (history[:, None] >= 0) & (history[:, None] < 3) & (t[:, None] < SPAN),
    )


def prefill_conv(x, weight, state, slots, span, lengths=None):
    sequences = len(x) // span
    output = torch.empty((len(x), 8192), device=x.device, dtype=x.dtype)
    _prefill_conv[(triton.cdiv(span, 8), 64, sequences)](
        x,
        weight,
        slots,
        lengths,
        state,
        output,
        span,
        x.stride(0),
        slots.stride(0),
        lengths is not None,
        8,
        128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
