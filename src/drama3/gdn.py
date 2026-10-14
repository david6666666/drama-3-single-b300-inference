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
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd


def prefill(q, k, v, decay, beta, state, slots, length):
    capacity = q.shape[1]
    chunks = (capacity + 63) // 64
    boundaries = torch.cat((torch.zeros_like(length), length))
    indices = torch.stack(
        (
            torch.zeros(chunks, device=q.device, dtype=torch.int64),
            torch.arange(chunks, device=q.device),
        ),
        dim=1,
    )
    offsets = torch.cat((torch.zeros_like(length), (length + 63) // 64))
    q, k, v, decay, beta = (tensor.contiguous() for tensor in (q, k, v, decay, beta))
    q, k = l2norm_fwd(q), l2norm_fwd(k)
    g = chunk_local_cumsum(decay, chunk_size=64, cu_seqlens=boundaries, chunk_indices=indices)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=boundaries, chunk_indices=indices
    )
    h = torch.empty((1, chunks, 32, 128, 128), dtype=k.dtype, device=k.device)
    updated = torch.empty_like(u)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[lambda meta: (128 // meta["BV"], 32)](
        k=k,
        v=u,
        w=w,
        v_new=updated,
        g=g,
        gk=None,
        h=h,
        initial_state=state,
        initial_state_indices=slots[:1],
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
    return output
