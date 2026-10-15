"""Diagnostic-only MXFP4 weights with BF16 activations on real MoE inputs.

Weight preparation follows FlashInfer's Apache-2.0 MoE test contract; the
permutation itself is supplied by FlashInfer. No serving path imports this.
"""

from functools import partial

import torch
import torch.nn.functional as F
import triton
from flashinfer import fp4_quantize, nvfp4_block_scale_interleave
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)

from drama3.kernels import combine_experts, linear, silu_mul


def prepare(weight, scale, gated):
    experts, rows, columns = weight.shape
    unit = torch.ones(1, device=weight.device)
    permutation = (
        _maybe_get_cached_w3_w1_permute_indices if gated else get_w2_permute_indices_with_cache
    )
    cache, packed, scales, restored = {}, [], [], []
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    for index in range(experts):
        dense = (
            (
                weight[index].float().reshape(rows // 128, 128, columns // 128, 128)
                * scale[index, :, None, :, None]
            )
            .reshape(rows, columns)
            .to(torch.bfloat16)
        )
        quantized, factors = fp4_quantize(
            dense, unit, sf_vec_size=32, sf_use_ue8m0=True, is_sf_swizzled_layout=False
        )
        quantized = quantized.view(torch.uint8).reshape(rows, columns // 2)
        factors = factors.view(torch.uint8).reshape(rows, columns // 32)
        nibbles = torch.stack((quantized & 15, quantized >> 4), dim=-1).reshape(rows, columns)
        restored.append(
            (
                values[nibbles.long()]
                * torch.exp2(factors.float() - 127).repeat_interleave(32, dim=-1)
            ).to(torch.bfloat16)
        )
        order = permutation(cache, quantized, 128, is_gated_act_gemm=True)
        scale_order = permutation(cache, factors, 128, num_elts_per_sf=16, is_gated_act_gemm=True)
        packed.append(quantized[order].contiguous())
        scales.append(nvfp4_block_scale_interleave(factors[scale_order].contiguous()))
    return (
        torch.stack(packed),
        torch.stack(scales).view(torch.float8_e4m3fn).reshape(experts, rows, columns // 32),
        torch.stack(restored),
    )


def benchmark(model, layer, x):
    if model.moe_layout != "major_k":
        raise ValueError("This diagnostic requires unshuffled source weights")
    experts = layer.experts
    up, up_scale, restored_up = prepare(experts.flashinfer_w13, experts.flashinfer_w13_scale, True)
    down, down_scale, restored_down = prepare(experts.w2_weight, experts.w2_scale_inv, False)
    unit = torch.ones(256, device=x.device)

    def run():
        projected = linear(x, layer.packed_input_projection)
        routed = trtllm_fp4_block_scale_moe(
            routing_logits=projected[:, 1024:1280].float(),
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=None,
            gemm1_weights=up,
            gemm1_weights_scale=up_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=down,
            gemm2_weights_scale=down_scale,
            gemm2_bias=None,
            output1_scale_scalar=unit,
            output1_scale_gate_scalar=unit,
            output2_scale_scalar=unit,
            num_experts=256,
            top_k=8,
            n_group=None,
            topk_group=None,
            intermediate_size=512,
            local_expert_offset=0,
            local_num_experts=256,
            routed_scaling_factor=1.0,
            routing_method_type=4,
            tune_max_num_tokens=4096,
        )
        if isinstance(routed, list):
            routed = routed[0]
        shared = linear(silu_mul(projected[:, :1024]), layer.shared_expert.down_proj.weight)
        return combine_experts(routed, shared, projected[:, 1280:1281])

    reference_fn = partial(model.moe, x, layer, True)
    reference, actual = reference_fn(), run()
    assert bool(actual.isfinite().all()), "Non-finite MXFP4 output"
    # Independently decode E2M1/UE8M0 bytes before expert GEMMs to distinguish
    # quantization error from a wrong shuffle or kernel contract.
    projected = linear(x, layer.packed_input_projection)
    probability = projected[:, 1024:1280].float().softmax(-1)
    weights, ids = probability.topk(8, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    routed = torch.zeros_like(x, dtype=torch.float32)
    routes = {}
    for token, row in enumerate(ids.cpu().tolist()):
        for rank, expert in enumerate(row):
            routes.setdefault(expert, []).append((token, rank))
    for expert, selections in routes.items():
        tokens, ranks = torch.tensor(selections, device=x.device).T
        values = F.linear(x[tokens], restored_up[expert]).float()
        up_values, gate = values.chunk(2, dim=-1)
        activation = (up_values * F.silu(gate)).to(torch.bfloat16)
        down_values = F.linear(activation, restored_down[expert]).float()
        routed.index_add_(0, tokens, down_values * weights[tokens, ranks, None])
    shared = linear(silu_mul(projected[:, :1024]), layer.shared_expert.down_proj.weight)
    restored_reference = combine_experts(routed.to(x.dtype), shared, projected[:, 1280:1281])
    parity = float(
        (
            (actual.float() - restored_reference.float()).square().mean()
            / restored_reference.float().square().mean()
        ).sqrt()
    )
    error = float(
        (
            (actual.float() - reference.float()).square().mean() / reference.float().square().mean()
        ).sqrt()
    )
    return dict(
        rrms=error,
        quantized_reference_rrms=parity,
        reference_us=triton.testing.do_bench_cudagraph(reference_fn) * 1000,
        mxfp4_us=triton.testing.do_bench_cudagraph(run) * 1000,
        description="Diagnostic only; lossy weight quantization; audio quality untested",
    )
