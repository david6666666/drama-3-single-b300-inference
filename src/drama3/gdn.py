"""Graph-safe variable-length prefill using SGLang/FLA's numerical kernels.

Static chunk descriptors avoid the upstream helper's device-to-host length
conversion. Kernels receive the true GPU sequence length, including for the
final recurrent-state update; zeroing padded beta alone is insufficient.
"""

import torch
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_kernel_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd


def decode(mixed, a, b, log, bias, state, output, slots):
    """Use state-dtype-specific tiles; arithmetic and state layout are unchanged."""
    tile = (16 if len(slots) <= 32 else 32) if state.dtype == torch.bfloat16 else 8
    fused_recurrent_gated_delta_rule_packed_decode_kernel[(128 // tile, len(slots) * 32)](
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=log,
        dt_bias=bias,
        o=output,
        h0=state,
        ht=state,
        ssm_state_indices=slots,
        scale=128**-0.5,
        stride_mixed_qkv_tok=mixed.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0),
        stride_indices_seq=slots.stride(0),
        H=16,
        HV=32,
        K=128,
        V=128,
        BK=128,
        BV=tile,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=1,
        num_stages=3,
    )


def prefill(q, k, v, decay, beta, state, slots, length):
    sequences, span = q.shape[:2]
    capacity = sequences * span
    chunks_per_sequence = (span + 63) // 64
    if sequences == 1:
        boundaries = torch.cat((torch.zeros_like(length), length))
        state_slots = slots[:1]
        sequence_count = 1
    else:
        # Each padded row is two independent variable-length sequences: its
        # real prefix and an ignored suffix in a scratch state slot. Masking
        # gates alone changes the chunked recurrence's final-state rounding.
        starts = torch.arange(sequences, device=q.device, dtype=torch.int64) * span
        boundaries = torch.cat(
            (
                torch.stack((starts, starts + length), dim=1).flatten(),
                torch.full((1,), capacity, device=q.device, dtype=torch.int64),
            )
        )
        real_slots = slots[::span]
        scratch = (real_slots + len(state) // 2) % len(state)
        state_slots = torch.stack((real_slots, scratch), dim=1).flatten()
        sequence_count = 2 * sequences
        q, k, v, decay, beta = (
            tensor.flatten(0, 1).unsqueeze(0) for tensor in (q, k, v, decay, beta)
        )
    chunks = sequence_count * chunks_per_sequence
    indices = torch.stack(
        (
            torch.arange(sequence_count, device=q.device).repeat_interleave(chunks_per_sequence),
            torch.arange(chunks_per_sequence, device=q.device).repeat(sequence_count),
        ),
        dim=1,
    )
    offsets = torch.arange(sequence_count + 1, device=q.device) * chunks_per_sequence
    q, k, v, decay, beta = (tensor.contiguous() for tensor in (q, k, v, decay, beta))
    q, k = l2norm_fwd(q), l2norm_fwd(k)
    g = chunk_local_cumsum(decay, chunk_size=64, cu_seqlens=boundaries, chunk_indices=indices)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=boundaries, chunk_indices=indices
    )
    h = torch.empty((1, chunks, 32, 128, 128), dtype=k.dtype, device=k.device)
    updated = torch.empty_like(u)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[
        lambda meta: (128 // meta["BV"], sequence_count * 32)
    ](
        k=k,
        v=u,
        w=w,
        v_new=updated,
        g=g,
        gk=None,
        h=h,
        initial_state=state,
        initial_state_indices=state_slots,
        stride_init_state=state.stride(0),
        cu_seqlens=boundaries,
        chunk_offsets=offsets,
        track_state=None,
        track_chunk_idx=None,
        stride_track_state=0,
        T=capacity,
        H=32,
        Hg=16,
        K=128,
        V=128,
        BT=64,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=True,
        TRACK_STATE=False,
        IS_VARLEN=True,
        NT_BUCKET=0,
        USE_EXP2=False,
    )
    output = torch.zeros_like(v)
    chunk_fwd_kernel_o[(2, chunks, 32)](
        q,
        k,
        updated,
        h,
        g,
        output,
        boundaries,
        indices,
        128**-0.5,
        T=capacity,
        H=32,
        Hg=16,
        K=128,
        V=128,
        BT=64,
        BK=128,
        BV=64,
        USE_G=True,
        IS_VARLEN=True,
        num_warps=4,
        num_stages=2,
    )
    return output.reshape(sequences, span, 32, 128)
