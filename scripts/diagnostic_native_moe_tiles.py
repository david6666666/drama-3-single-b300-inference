"""Tune output-column tiles of unchanged native FP8 expert GEMMs on real inputs.

Only in-memory launch constants of the pinned Fish kernel are varied. Source,
weights, 128-channel FP8 scales, all top-8 experts and rounding boundaries stay
unchanged. The serving model is not patched by this diagnostic.
"""

import json
from functools import partial
from itertools import product

import torch
import triton
from fish_speech.models.text2semantic.fish_omni import fp8_moe

from drama3.kernels import linear, native_moe, route


def rrms(actual, expected):
    return float(
        (
            (actual.float() - expected.float()).square().mean()
            / expected.float().square().mean().clamp_min(1e-12)
        ).sqrt()
    )


@torch.inference_mode()
def benchmark(model, activations, output):
    saved = fp8_moe.GEMM_BLOCK_N, fp8_moe.GEMM_NUM_WARPS, fp8_moe.GEMM_NUM_STAGES
    results = []
    try:
        for index, x in activations.items():
            layer = model.backbone.layers[index].mlp
            router = (
                linear(x, layer.packed_input_projection)[:, 1024:1280]
                if model.packed_moe_projection
                else linear(x, layer.gate.weight)
            )

            def native(x=x, router=router, experts=layer.experts):
                weights, ids = route(router)
                return native_moe(x, ids, weights, experts)

            (fp8_moe.GEMM_BLOCK_N, fp8_moe.GEMM_NUM_WARPS, fp8_moe.GEMM_NUM_STAGES) = saved
            reference = native()
            current = partial(model.routed_moe, x, router, layer.experts, True)
            current_output = current()
            current_us = triton.testing.do_bench_cudagraph(current) * 1000
            reference_us = triton.testing.do_bench_cudagraph(native) * 1000
            for tile, warps, stages in product((128, 256, 512), (4, 8), (2, 3)):
                fp8_moe.GEMM_BLOCK_N = tile
                fp8_moe.GEMM_NUM_WARPS = warps
                fp8_moe.GEMM_NUM_STAGES = stages
                row = dict(
                    layer=index,
                    batch=len(x),
                    block_n=tile,
                    warps=warps,
                    stages=stages,
                    current_routed_us=current_us,
                    reference_native_us=reference_us,
                    original_native_vs_current_rrms=rrms(reference, current_output),
                )
                try:
                    actual = native()
                    error = rrms(actual, reference)
                    row.update(rrms=error, passed=bool(actual.isfinite().all()) and error < 0.003)
                    if row["passed"]:
                        row["native_us"] = triton.testing.do_bench_cudagraph(native) * 1000
                except triton.OutOfResources as error:
                    row.update(passed=False, resource_rejection=str(error))
                results.append(row)
                output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                if not row["passed"] and "resource_rejection" not in row:
                    raise AssertionError("Native tile changed numerical output beyond fixed bound")
    finally:
        (fp8_moe.GEMM_BLOCK_N, fp8_moe.GEMM_NUM_WARPS, fp8_moe.GEMM_NUM_STAGES) = saved
