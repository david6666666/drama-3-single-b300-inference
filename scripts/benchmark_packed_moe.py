"""Validate and time a shared MoE/router input projection without quantization."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    torch.manual_seed(42)
    results = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 256):
        x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        for index in (0, 19, 39):
            layer = model.backbone.layers[index].mlp
            fn = partial(model.moe, x, layer, True)
            model.packed_moe_projection = False
            expected = fn()
            baseline = triton.testing.do_bench_cudagraph(fn) * 1000
            model.packed_moe_projection = True
            actual = fn()
            error = float(
                (
                    (actual.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            assert bool(actual.isfinite().all()) and error < 0.01, (batch, index, error)
            elapsed = triton.testing.do_bench_cudagraph(fn) * 1000
            result = dict(
                batch=batch, layer=index, rrms=error, baseline_us=baseline, packed_us=elapsed
            )
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
