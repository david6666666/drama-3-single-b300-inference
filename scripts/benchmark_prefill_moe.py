"""Compare complete MoE math at prefill sizes, including native FP8 semantics."""

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
    for batch in (16, 64, 128, 512, 2048, 4096):
        x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        for index in (0, 19, 39):
            layer = model.backbone.layers[index].mlp
            reference = layer(x)
            for name, fn in (
                ("native", partial(model.moe, x, layer, False)),
                ("flashinfer", partial(model.moe, x, layer, True)),
            ):
                output = fn()
                rrms = float(
                    (
                        (output.float() - reference.float()).square().mean()
                        / reference.float().square().mean()
                    ).sqrt()
                )
                assert bool(output.isfinite().all()), (batch, index, name)
                result = dict(
                    batch=batch,
                    layer=index,
                    backend=name,
                    rrms=rrms,
                    microseconds=triton.testing.do_bench_cudagraph(fn) * 1000,
                )
                results.append(result)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
