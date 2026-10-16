"""Diagnostic MXFP8 experts; no runtime dispatch imports this module.

Quantize and shuffle through FlashInfer's public helpers. Independently decode
the unshuffled E4M3/UE8M0 bytes before comparing the routed expert arithmetic.
"""

from functools import partial

import torch
import torch.nn.functional as F
import triton
from flashinfer import (
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from flashinfer.fused_moe.core import Fp8QuantizationType, WeightLayout

from drama3.kernels import combine_experts, linear, silu_mul


def restore(quantized, scales):
    rows, columns = quantized.shape
    factors = scales.view(torch.uint8).reshape(rows, columns // 32)
    return (quantized.float() * torch.exp2(factors.float() - 127).repeat_interleave(32, -1)).to(
        torch.bfloat16
    )


def prepare(weight, scale, gated):
    experts, rows, columns = weight.shape
    packed, factors, restored = [], [], []
    for expert in range(experts):
        dense = (
            (
                weight[expert].float().reshape(rows // 128, 128, columns // 128, 128)
                * scale[expert, :, None, :, None]
            )
            .reshape(rows, columns)
            .to(torch.bfloat16)
        )
        quantized, scales = mxfp8_quantize(dense, False)
        scales = scales.view(torch.uint8).reshape(rows, columns // 32)
        restored.append(restore(quantized, scales))
        if gated:
            quantized = reorder_rows_for_gated_act_gemm(quantized)
            scales = reorder_rows_for_gated_act_gemm(scales)
        packed.append(shuffle_matrix_a(quantized.view(torch.uint8), 128).contiguous())
        factors.append(shuffle_matrix_sf_a(scales, 128).contiguous())
    return (
        torch.stack(packed).view(torch.float8_e4m3fn),
        torch.stack(factors),
        torch.stack(restored),
    )


def benchmark(model, layer, x):
    if model.moe_layout != "major_k":
        raise ValueError("Diagnostic requires unshuffled source weights")
    experts = layer.experts
    up, up_scale, restored_up = prepare(experts.flashinfer_w13, experts.flashinfer_w13_scale, True)
    down, down_scale, restored_down = prepare(experts.w2_weight, experts.w2_scale_inv, False)

    def run():
        projected = linear(x, layer.packed_input_projection)
        q, scale = mxfp8_quantize(x, False)
        routed = trtllm_fp8_block_scale_moe(
            routing_logits=projected[:, 1024:1280].float(),
            routing_bias=None,
            hidden_states=q,
            hidden_states_scale=scale.view(torch.uint8).reshape(len(x), -1),
            gemm1_weights=up,
            gemm1_weights_scale=up_scale,
            gemm2_weights=down,
            gemm2_weights_scale=down_scale,
            num_experts=256,
            top_k=8,
            n_group=None,
            topk_group=None,
            intermediate_size=512,
            local_expert_offset=0,
            local_num_experts=256,
            routed_scaling_factor=1.0,
            routing_method_type=4,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            tune_max_num_tokens=4096,
        )
        if isinstance(routed, list):
            routed = routed[0]
        shared = linear(silu_mul(projected[:, :1024]), layer.shared_expert.down_proj.weight)
        return combine_experts(routed, shared, projected[:, 1280:1281])

    reference_fn = partial(model.moe, x, layer, True)
    expected, actual = reference_fn(), run()
    assert bool(actual.isfinite().all()), "Nonfinite MXFP8 output"
    projected = linear(x, layer.packed_input_projection)
    probabilities = projected[:, 1024:1280].float().softmax(-1)
    weights, ids = probabilities.topk(8, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    q, scales = mxfp8_quantize(x, False)
    dequantized = restore(q, scales)
    routed = torch.zeros_like(x, dtype=torch.float32)
    routes = {}
    for token, row in enumerate(ids.cpu().tolist()):
        for rank, expert in enumerate(row):
            routes.setdefault(expert, []).append((token, rank))
    for expert, selections in routes.items():
        tokens, ranks = torch.tensor(selections, device=x.device).T
        values = F.linear(dequantized[tokens], restored_up[expert]).float()
        up_values, gate = values.chunk(2, -1)
        activation = (up_values * F.silu(gate)).to(torch.bfloat16)
        aq, ascales = mxfp8_quantize(activation, False)
        output = F.linear(restore(aq, ascales), restored_down[expert]).float()
        routed.index_add_(0, tokens, output * weights[tokens, ranks, None])
    shared = linear(silu_mul(projected[:, :1024]), layer.shared_expert.down_proj.weight)
    independent = combine_experts(routed.to(x.dtype), shared, projected[:, 1280:1281])

    def rrms(reference):
        return float(
            (
                (actual.float() - reference.float()).square().mean()
                / reference.float().square().mean()
            ).sqrt()
        )

    parity, error = rrms(independent), rrms(expected)
    result = dict(
        rrms=error,
        quantized_reference_rrms=parity,
        passed=parity < 0.03 and error < 0.1,
        description="Diagnostic quantization change; no audio quality acceptance",
    )
    if result["passed"]:
        result.update(
            reference_us=triton.testing.do_bench_cudagraph(reference_fn) * 1000,
            mxfp8_us=triton.testing.do_bench_cudagraph(run) * 1000,
        )
    return result
