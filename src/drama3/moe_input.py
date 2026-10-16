"""Fuse unchanged native FP8 activation preparation and router conversion."""

import torch
import triton
import triton.language as tl


@triton.jit
def prepare_kernel(
    X,
    ROUTER,
    Q,
    S,
    LOGITS,
    ROW_SLOTS,
    B: tl.constexpr,
    SX: tl.constexpr,
    SR: tl.constexpr,
    PADDING_START: tl.constexpr,
):
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
    router_row = row
    if PADDING_START:
        slot = tl.load(ROW_SLOTS + row)
        router_row = tl.where(slot >= PADDING_START, 0, row)
    logits = tl.load(ROUTER + router_row * SR + expert).to(tl.float32)
    tl.store(LOGITS + row * 256 + expert, logits)


def prepare_moe_input(x, router, *, row_slots=None, padding_start=None):
    """Optionally reuse row 0 routing for explicitly reserved padding slots.

    Callers enabling this path must put a real request in row 0. Activation
    quantization stays per-row and unchanged, including for padding rows.
    Without slot metadata, generic callers retain their original routing.
    """
    if x.shape[1] != 2048 or router.shape != (len(x), 256):
        raise ValueError("Expected native Drama 3 hidden and full 256-expert router shapes")
    if x.stride(1) != 1 or router.stride(1) != 1:
        raise ValueError("Channel strides must be contiguous")
    if (row_slots is None) != (padding_start is None):
        raise ValueError("Reserved padding routing requires both row_slots and padding_start")
    if row_slots is not None:
        if (
            row_slots.shape != (len(x),)
            or row_slots.stride(0) != 1
            or row_slots.device != x.device
            or row_slots.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Expected one contiguous integer slot per MoE row on the same device")
        if not isinstance(padding_start, int) or padding_start <= 0:
            raise ValueError("padding_start must be a positive reserved-slot boundary")
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((16, len(x)), device=x.device, dtype=torch.float32)
    logits = torch.empty(router.shape, device=x.device, dtype=torch.float32)
    prepare_kernel[(len(x),)](
        x,
        router,
        q,
        scales,
        logits,
        row_slots,
        len(x),
        x.stride(0),
        router.stride(0),
        padding_start or 0,
        num_warps=4,
    )
    return q, scales, logits
