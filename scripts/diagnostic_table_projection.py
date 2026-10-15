"""Compare fused small-batch tower projections to the unchanged native formula."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.kernels import table_projection
from drama3.model import DramaModel
from drama3.table_projection_batched import table_projection_batched


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=2, max_length=512)
    torch.manual_seed(42)
    results = []
    for batch in (2, 4, 8):
        for index, layer in enumerate([*model.tower.shared_layers, *model.tower.residual_layers]):
            if not index:
                continue
            hidden = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
            codes = torch.randint(-1, 4097, (batch,), device="cuda")
            semantic = torch.arange(batch, device="cuda") % 2 == 0
            native = partial(table_projection, hidden, layer, codes, semantic)
            expected = native()
            native_us = triton.testing.do_bench_cudagraph(native) * 1000
            for rows in (1, 2, 4, 8):
                candidate = partial(table_projection_batched, hidden, layer, codes, semantic, rows)
                actual = candidate()
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean().clamp_min(1e-12)
                    ).sqrt()
                )
                passed = bool(actual.isfinite().all()) and error < 0.003
                result = dict(
                    batch=batch,
                    layer=index,
                    rows=rows,
                    rrms=error,
                    passed=passed,
                    native_us=native_us,
                )
                results.append(result)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                assert passed, result
                result["fused_us"] = triton.testing.do_bench_cudagraph(candidate) * 1000
                args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(min(results[-4:], key=lambda r: r["fused_us"])), flush=True)


if __name__ == "__main__":
    main()
