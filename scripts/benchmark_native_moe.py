"""Check and time exact-quantization MoE backends on loaded Drama 3 weights."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from drama3.kernels import native_moe, small_batch_moe
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    torch.manual_seed(42)
    result = []
    for batch in args.batches:
        x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        for index in (0, 19, 39):
            layer = model.backbone.layers[index].mlp
            probability = F.linear(x, layer.gate.weight).float().softmax(-1)
            weights, ids = probability.topk(8, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
            expected = layer.experts(x, ids, weights)
            for name, fn in (
                ("reference", partial(layer.experts, x, ids, weights)),
                ("gemv", partial(small_batch_moe, x, ids, weights, layer.experts)),
                ("native", partial(native_moe, x, ids, weights, layer.experts)),
            ):
                if name == "gemv" and batch > 64:
                    continue
                actual = fn()
                error = (
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    )
                    .sqrt()
                    .item()
                )
                assert error < (0.003 if name == "native" else 0.03), (batch, index, name, error)
                elapsed = triton.testing.do_bench_cudagraph(fn)
                row = {
                    "batch": batch,
                    "layer": index,
                    "backend": name,
                    "relative_rmse": error,
                    "microseconds": elapsed * 1000,
                }
                result.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
