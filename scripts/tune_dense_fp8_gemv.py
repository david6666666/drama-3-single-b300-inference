"""Test whether blockwise scale reduction makes the opt-in FP8 GEMV useful.

This is a diagnostic, not a quality or end-to-end performance result.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.experimental_fp8 import WEIGHTS, _gemv, _gemv_grouped, register
from drama3.kernels import linear
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    matrices = {
        "gdn_input": model.backbone.layers[0].linear_attn.packed_input,
        "gdn_output": model.backbone.layers[0].linear_attn.out_proj.weight,
        "tower_ffn": model.tower.shared_layers[0].transformer_layer.feed_forward.packed_gate_up,
    }
    torch.manual_seed(42)
    results = []
    for name, weight in matrices.items():
        n, k = weight.shape
        x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
        expected = linear(x, weight)
        native_us = triton.testing.do_bench_cudagraph(partial(linear, x, weight)) * 1000
        register(weight)
        _, quantized, scales = WEIGHTS[weight.data_ptr()]
        output = torch.empty_like(expected)
        for kernel in (_gemv, _gemv_grouped):
            for rows in (1, 2, 4, 8, 16):
                for warps in (4, 8):
                    fn = partial(
                        kernel[(triton.cdiv(n, rows),)],
                        x,
                        quantized,
                        scales,
                        output,
                        n,
                        k,
                        rows,
                        num_warps=warps,
                    )
                    fn()
                    error = float(
                        (
                            (output.float() - expected.float()).square().mean()
                            / expected.float().square().mean()
                        ).sqrt()
                    )
                    elapsed = triton.testing.do_bench_cudagraph(fn) * 1000
                    result = dict(
                        matrix=name,
                        kernel=kernel.__name__,
                        rows=rows,
                        warps=warps,
                        rrms=error,
                        native_us=native_us,
                        fp8_us=elapsed,
                        lossy=True,
                    )
                    results.append(result)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
