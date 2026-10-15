"""BF16 single-request gating and projection, retaining activation rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def gated_gemv_kernel(
    X,
    GATE,
    W,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    MODE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_K)
    x = tl.load(X + columns, columns < K, other=0).to(tl.float32)
    gate = tl.load(GATE + columns, columns < K, other=0).to(tl.float32)
    if MODE == "swiglu":
        activation = (gate * tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
    else:
        activation = tl.sigmoid(gate).to(X.dtype.element_ty).to(tl.float32)
    activated = (x * activation).to(X.dtype.element_ty).to(tl.float32)
    weights = tl.load(
        W + rows[:, None] * K + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K),
        other=0,
    ).to(tl.float32)
    output = tl.sum(weights * activated[None, :], 1)
    tl.store(Y + rows, output, rows < N)


def gated_gemv(x, gate, weight, mode, rows=None, warps=None):
    if x.shape[0] != 1 or x.dtype != torch.bfloat16:
        raise ValueError("Fused gating requires one BF16 row")
    if mode not in ("swiglu", "sigmoid"):
        raise ValueError("Expected swiglu or sigmoid")
    n, k = weight.shape
    rows = rows or (2 if k == 4096 else 4)
    warps = warps or (4 if k <= 4096 else 8)
    output = torch.empty((1, n), dtype=x.dtype, device=x.device)
    gated_gemv_kernel[(triton.cdiv(n, rows),)](
        x,
        gate,
        weight,
        output,
        n,
        k,
        mode,
        rows,
        triton.next_power_of_2(k),
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return output
