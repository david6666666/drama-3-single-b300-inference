"""Check partitioned sampler support, deterministic draws, and graph latency."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton

from drama3.kernels import sample, sample_partitioned


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    result = []
    for batch in (1, 8, 64):
        for width in (4096, 4097):
            for scale in (1.0, 5.0, 10.0):
                logits = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16) * scale
                step = torch.zeros((), dtype=torch.int64, device="cuda")
                for top_p in (0.9, 0.95):
                    for draw in range(256):
                        step.fill_(draw)
                        expected = sample(logits, step, 0, top_p)
                        actual = sample_partitioned(logits, step, 0, top_p)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                old = triton.testing.do_bench_cudagraph(partial(sample, logits, step, 0))
                new = triton.testing.do_bench_cudagraph(
                    partial(sample_partitioned, logits, step, 0)
                )
                row = {
                    "batch": batch,
                    "width": width,
                    "scale": scale,
                    "passed": True,
                    "original_us": old * 1000,
                    "partitioned_us": new * 1000,
                }
                result.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
