"""Opt-in native BF16 table projection with a fused small-batch epilogue."""

import torch
import triton
import triton.language as tl


@triton.jit
def projection(X, W, NORM, TABLE, CODE, SEMANTIC, OUT, EPS: tl.constexpr, ROWS: tl.constexpr):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, 2048)
    x = tl.load(X + batch * 2048 + columns).to(tl.float32)
    norm = 1.0 + tl.load(NORM + columns).to(tl.float32)
    normalized = (x * tl.rsqrt(tl.sum(x * x, 0) / 2048 + EPS) * norm).to(X.dtype.element_ty)
    weight = tl.load(W + rows[:, None] * 2048 + columns[None, :]).to(tl.float32)
    partial = tl.sum(weight * normalized.to(tl.float32)[None, :], 1)
    code = tl.minimum(tl.maximum(tl.load(CODE + batch), 0), 4095)
    semantic = tl.load(SEMANTIC + batch)
    embedding = tl.load(TABLE + code * 2048 + rows)
    tl.store(OUT + batch * 2048 + rows, partial + tl.where(semantic, embedding, 0.0))


def table_projection_batched(hidden, layer, codes, semantic, rows=4):
    if not hidden.is_contiguous() or hidden.shape[1] != 2048 or hidden.dtype != torch.bfloat16:
        raise ValueError("The diagnostic requires contiguous BF16 hidden states of width 2048")
    output = torch.empty_like(hidden)
    projection[(2048 // rows, len(hidden))](
        hidden,
        layer.hidden_projection,
        layer.hnorm.weight,
        layer.code_table,
        codes,
        semantic,
        output,
        layer.hnorm.eps,
        rows,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
