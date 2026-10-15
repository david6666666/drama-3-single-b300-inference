"""Tune existing FP8 expert GEMV tiles without altering quantization boundaries."""

from functools import partial

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    quant_kernel,
    quantize_groups,
    silu_mul_quant_kernel,
)

from drama3.kernels import expert_down_kernel, expert_up_kernel, linear, route, small_batch_moe


def benchmark(model, layer, x):
    weights, ids = route(linear(x, layer.gate.weight))
    experts = layer.experts
    q, scales = quantize_groups(quant_kernel, x, 2048)
    up = torch.empty(len(x) * 8, 1024, device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    results = []
    for rows in (1, 2, 4, 8, 16):
        for warps in (4, 8):
            fn = partial(
                expert_up_kernel[(len(x), 8, 1024 // rows)],
                q,
                scales,
                experts.w13_weight,
                experts.w13_scale_inv,
                ids,
                up,
                rows,
                num_warps=warps,
            )
            fn()
            if rows == 1 and warps == 4:
                expected = up.clone()
            error = float(
                (
                    (up.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            assert error < 0.003, error
            results.append(
                dict(
                    stage="up",
                    rows=rows,
                    warps=warps,
                    rrms=error,
                    hot_us=triton.testing.do_bench_cudagraph(fn) * 1000,
                    cold_us=triton.testing.do_bench(fn) * 1000,
                )
            )
    activated, activation_scales = quantize_groups(silu_mul_quant_kernel, up, 512)
    for rows in (1, 2, 4, 8):
        for warps in (4, 8):
            fn = partial(
                expert_down_kernel[(len(x), 2048 // rows)],
                activated,
                activation_scales,
                experts.w2_weight,
                experts.w2_scale_inv,
                ids,
                weights,
                out,
                rows,
                num_warps=warps,
            )
            fn()
            if rows == 1 and warps == 4:
                expected = out.clone()
            error = float(
                (
                    (out.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            assert error < 0.003, error
            results.append(
                dict(
                    stage="down",
                    rows=rows,
                    warps=warps,
                    rrms=error,
                    hot_us=triton.testing.do_bench_cudagraph(fn) * 1000,
                    cold_us=triton.testing.do_bench(fn) * 1000,
                )
            )
    reference_fn = partial(small_batch_moe, x, ids, weights, experts)
    return dict(
        configurations=results,
        baseline_hot_us=triton.testing.do_bench_cudagraph(reference_fn) * 1000,
        baseline_cold_us=triton.testing.do_bench(reference_fn) * 1000,
    )
