"""Skip unused packed chunks while reusing SGLang/FLA's native arithmetic.

The imported Triton functions retain their upstream implementations and
copyrights (Songlin Yang and Yu Zhang, 2023-2025). These wrappers only guard
invalid chunks; they do not replace the matrix arithmetic or BF16 boundaries.
"""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_kkt_solve_kernel,
)
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_kernel_o
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd_kernel


@triton.jit(do_not_specialize=["T"])
def compact_chunk_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    chunk = tl.program_id(1)
    sequence = tl.load(chunk_indices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(chunk_indices + chunk * 2 + 1).to(tl.int32)
    length = tl.load(cu_seqlens + sequence + 1) - tl.load(cu_seqlens + sequence)
    if local_chunk * BT >= length:
        return
    chunk_fwd_kernel_o(
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T,
        H,
        Hg,
        K,
        V,
        BT,
        BK,
        BV,
        USE_G,
        IS_VARLEN,
    )


@triton.jit(do_not_specialize=["T"])
def compact_recompute(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    chunk = tl.program_id(0)
    sequence = tl.load(chunk_indices + chunk * 2).to(tl.int32)
    local_chunk = tl.load(chunk_indices + chunk * 2 + 1).to(tl.int32)
    length = tl.load(cu_seqlens + sequence + 1) - tl.load(cu_seqlens + sequence)
    if local_chunk * BT >= length:
        return
    recompute_w_u_fwd_kernel(
        k,
        v,
        beta,
        w,
        u,
        A,
        g,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        Hg,
        K,
        V,
        BT,
        BK,
        BV,
        IS_VARLEN,
    )


def compact_intra(k, v, beta, g, boundaries, indices):
    """Keep FLA's KKT solve and recomputation with guarded compact launches."""
    batch, tokens, query_heads, key_size = k.shape
    heads, value_size = v.shape[-2:]
    a = torch.zeros(batch, tokens, heads, 64, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(len(indices), batch * heads)](
        k=k,
        g=g,
        beta=beta,
        A=a,
        cu_seqlens=boundaries,
        chunk_indices=indices,
        T=tokens,
        H=heads,
        Hg=query_heads,
        K=key_size,
        BT=64,
        BC=16,
    )
    w = k.new_empty(batch, tokens, heads, key_size)
    u = torch.empty_like(v)
    compact_recompute[(len(indices), batch * heads)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=a,
        g=g,
        cu_seqlens=boundaries,
        chunk_indices=indices,
        T=tokens,
        H=heads,
        Hg=query_heads,
        K=key_size,
        V=value_size,
        BT=64,
        BK=64,
        BV=64,
        IS_VARLEN=True,
        num_warps=4,
        num_stages=3,
    )
    return w, u
