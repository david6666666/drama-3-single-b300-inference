"""Measure the explicitly lossy dense FP8 prototype against native BF16 projections."""

import argparse
import json
from pathlib import Path

import torch
import triton

from drama3.experimental_fp8 import linear as fp8_linear
from drama3.experimental_fp8 import register
from drama3.kernels import linear
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    matrices = {
        "gdn_input": model.backbone.layers[0].linear_attn.packed_input,
        "gdn_output": model.backbone.layers[0].linear_attn.out_proj.weight,
        "tower_ffn": model.tower.shared_layers[0].transformer_layer.feed_forward.packed_gate_up,
    }
    results = []
    torch.manual_seed(42)
    for name, weight in matrices.items():
        for batch in (1, 2, 8, 32, 64, 256):
            from drama3.experimental_fp8 import WEIGHTS

            WEIGHTS.pop(weight.data_ptr(), None)
            x = torch.randn(batch, weight.shape[1], device="cuda", dtype=torch.bfloat16)
            expected = linear(x, weight)
            native_ms = triton.testing.do_bench_cudagraph(
                lambda x=x, weight=weight: linear(x, weight)
            )
            register(weight)
            actual = fp8_linear(x, weight)
            error = float(
                (
                    (actual.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            elapsed = triton.testing.do_bench_cudagraph(
                lambda x=x, weight=weight: fp8_linear(x, weight)
            )
            row = {
                "matrix": name,
                "batch": batch,
                "relative_rmse": error,
                "native_us": native_ms * 1000,
                "fp8_us": elapsed * 1000,
                "lossy": True,
            }
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
