"""Check causal codec padding and graph replay against eager waveform decoding."""

import argparse
import json
from pathlib import Path

import torch

from drama3.codec import CodecGraph, load_codec


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    codec = load_codec(args.checkpoint, args.fish_source)
    results = []
    for batch in (1, 4):
        for length in (10, 20, 32, 64, 96):
            graph = CodecGraph(codec, batch, length)
            codes = torch.randint(1024, (batch, 10, length), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = codec.from_indices(codes)
            actual = graph.run(codes)
            error = (
                (
                    (actual.float() - expected.float()).square().mean()
                    / expected.float().square().mean().clamp_min(1e-12)
                )
                .sqrt()
                .item()
            )
            row = {"batch": batch, "length": length, "relative_rmse": error, "pass": error < 0.03}
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    assert all(row["pass"] for row in results), "Codec padding failed; inspect all lengths"


if __name__ == "__main__":
    main()
