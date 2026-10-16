"""One backbone pass for disjoint decode slots and packed prompt slots.

Rows are ordered as a fixed decode prefix followed by a packed prefill suffix.
Dense projections and the existing full top-8 MoE process all rows together;
attention and recurrent updates retain their existing per-phase operators.
The caller owns metadata validation and resets only prefill sequence states.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from drama3.gdn import decode as gdn_decode
from drama3.gdn import prefill as gdn_prefill
from drama3.kernels import (
    add_norm,
    combine_experts_add_norm,
    combine_experts_add_norm_inverse,
    conv_update,
    gated_norm,
    linear,
    prepare_attention,
)
from drama3.packed_prefill import PackedLayout, packed_conv

if TYPE_CHECKING:
    from drama3.model import AttentionBatch, DramaModel


def mixed_attention(
    model: DramaModel,
    x: torch.Tensor,
    layer,
    index: int,
    positions: torch.Tensor,
    slots: torch.Tensor,
    batch: AttentionBatch,
    layout: PackedLayout,
    decode_rows: int,
) -> torch.Tensor:
    """Share projections while retaining decode and packed causal attention."""
    tower = index >= 40
    projected = linear(x, layer.wqkv.weight if tower else layer.packed_qkv)
    keys, values = model.kv[index]
    decode_queries = prepare_attention(
        projected[:decode_rows],
        layer.q_norm.weight,
        layer.k_norm.weight,
        model.cosine,
        model.sine,
        positions[:decode_rows],
        slots[:decode_rows],
        keys,
        values,
        layer.q_norm.eps,
        tower,
    )
    decode_output = batch.run(decode_queries, keys, values)

    if tower:
        q, gate, k, v = projected.split((4096, 4096, 512, 512), dim=-1)
    else:
        qgate, k, v = projected.split((8192, 512, 512), dim=-1)
        q, gate = qgate.view(-1, 16, 512).chunk(2, dim=-1)
        gate = gate.reshape(-1, 4096)
    prefill_positions = positions[decode_rows:]
    prefill_slots = slots[decode_rows:]
    # Keep the existing prefill BF16 normalization/rotation boundaries. The
    # decode prefix above continues to use the established fused preparation.
    q = model.rotate(
        model.norm(q[decode_rows:].reshape(-1, 256), layer.q_norm).reshape(-1, 16, 256),
        prefill_positions,
        tower,
    )
    k = model.rotate(
        model.norm(k[decode_rows:].reshape(-1, 256), layer.k_norm).reshape(-1, 2, 256),
        prefill_positions,
        tower,
    )
    v = v[decode_rows:].reshape(-1, 2, 256)
    keys[prefill_slots, prefill_positions] = k
    values[prefill_slots, prefill_positions] = v
    prefill_output = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        layout.boundaries,
        layout.boundaries,
        layout.max_length,
        layout.max_length,
        0.0,
        True,
        False,
    )[0]
    output = torch.cat(
        (decode_output.reshape(decode_rows, 4096), prefill_output.reshape(-1, 4096)), dim=0
    )
    return linear(output * gate.sigmoid(), layer.wo.weight if tower else layer.o_proj.weight)


def mixed_linear_attention(
    model: DramaModel,
    x: torch.Tensor,
    layer,
    index: int,
    slots: torch.Tensor,
    layout: PackedLayout,
    decode_rows: int,
) -> torch.Tensor:
    """Share GDN projections, with disjoint recurrent state updates per phase."""
    projected = linear(x, layer.packed_input)
    mixed, z, b, a = projected.split((8192, 4096, 32, 32), dim=-1)
    conv, state = model.states[index]
    decode_slots = slots[:decode_rows]
    decode_mixed = conv_update(mixed[:decode_rows], conv, layer.conv1d.weight, decode_slots)
    decode_output = torch.empty((decode_rows, 1, 32, 128), dtype=x.dtype, device=x.device)
    gdn_decode(
        decode_mixed,
        a[:decode_rows],
        b[:decode_rows],
        layer.A_log_fp32,
        layer.dt_bias_fp32,
        state,
        decode_output,
        decode_slots,
    )

    prefill_mixed = packed_conv(mixed[decode_rows:], layer.conv1d.weight, conv, layout)
    q, k, v = prefill_mixed.split((2048, 2048, 4096), dim=-1)
    decay = -layer.A_log.float().exp() * F.softplus(a[decode_rows:].float() + layer.dt_bias.float())
    beta = b[decode_rows:].float().sigmoid()
    prefill_output = gdn_prefill(
        q.reshape(1, -1, 16, 128),
        k.reshape(1, -1, 16, 128),
        v.reshape(1, -1, 32, 128),
        decay.reshape(1, -1, 32),
        beta.reshape(1, -1, 32),
        state,
        slots[decode_rows:],
        None,
        backend=model.prefill_gdn,
        packed_layout=layout,
    )
    if os.environ.get("DRAMA3_FUSED_MIXED_LAYOUT") == "1":
        from drama3.mixed_ops import mixed_gated_norm

        output = mixed_gated_norm(
            decode_output, prefill_output, z, layer.norm.weight, layer.norm.eps
        )
    else:
        output = torch.cat(
            (decode_output.reshape(decode_rows, 4096), prefill_output.reshape(-1, 4096)), dim=0
        )
        output = gated_norm(
            output.reshape(-1, 1, 32, 128), z.contiguous(), layer.norm.weight, layer.norm.eps
        ).reshape(-1, 4096)
    return linear(output, layer.out_proj.weight)


def mixed_backbone(
    model: DramaModel,
    frames: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    batch: AttentionBatch,
    layout: PackedLayout,
    decode_rows: int,
    moe_row_reuse=None,
) -> torch.Tensor:
    """Execute decode rows first and packed prefill rows second in one backbone.

    ``layout`` boundaries start at zero within the prefill suffix. Every slot in
    ``layout.sequence_slots``, including empty sequences and trailing padding,
    must be disjoint from the decode slots. The graph wrapper validates this on
    CPU and resets these prefill slots before this function; decode state is
    retained. This function does not sample or execute the audio tower.
    """
    if not 0 < decode_rows < len(frames):
        raise ValueError("Mixed backbone requires nonempty decode and prefill segments")
    if (
        len(batch.slots) != decode_rows
        or len(positions) != len(frames)
        or len(slots) != len(frames)
    ):
        raise ValueError("Mixed backbone row metadata does not match its fixed segments")
    if model.moe_shared_stream is not None:
        raise ValueError("Mixed backbone requires DRAMA3_MOE_SHARED_OVERLAP=0")

    if moe_row_reuse is not None:
        if not model.fused_moe_residual:
            raise ValueError("Mixed row reuse requires the native fused MoE residual")
        keep, inverse, compact_slots = moe_row_reuse
        # Row metadata is already populated, and slots stay fixed throughout all 40 layers.
        torch.index_select(slots, 0, keep, out=compact_slots)

    def combine(parts, residual, norm):
        if moe_row_reuse is None:
            return combine_experts_add_norm(*parts, residual, norm.weight, norm.eps)
        return combine_experts_add_norm_inverse(*parts, inverse, residual, norm.weight, norm.eps)

    hidden = model.tower.embed_backbone(frames, model.backbone.embed_tokens(frames[:, 0]))
    pending = None
    for index, layer in enumerate(model.backbone.layers):
        normalized = (
            model.norm(hidden, layer.input_layernorm)
            if pending is None
            else combine(pending, hidden, layer.input_layernorm)
            if model.fused_moe_residual
            else add_norm(pending, hidden, layer.input_layernorm.weight, layer.input_layernorm.eps)
        )
        if layer.layer_type == "full_attention":
            mixed = mixed_attention(
                model,
                normalized,
                layer.self_attn,
                index,
                positions,
                slots,
                batch,
                layout,
                decode_rows,
            )
        else:
            mixed = mixed_linear_attention(
                model, normalized, layer.linear_attn, index, slots, layout, decode_rows
            )
        normalized = add_norm(
            mixed,
            hidden,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
        )
        pending = (
            model.moe_parts(
                normalized if moe_row_reuse is None else normalized.index_select(0, keep),
                layer.mlp,
                decode=True,
                row_slots=slots if moe_row_reuse is None else compact_slots,
            )
            if model.fused_moe_residual
            else model.moe(normalized, layer.mlp, decode=True, row_slots=slots)
        )
    if model.fused_moe_residual:
        return combine(pending, hidden, model.backbone.norm)
    return add_norm(pending, hidden, model.backbone.norm.weight, model.backbone.norm.eps)
