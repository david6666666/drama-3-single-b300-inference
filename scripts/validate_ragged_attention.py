"""Feasibility probe for graph-safe packed prefill attention, including GQA."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    capacity = 1536
    q = torch.randn(capacity, 16, 256, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn(2, capacity, 2, 256, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.zeros(9, device="cuda", dtype=torch.int32)

    def run():
        return torch.ops.aten._flash_attention_forward(
            q, k, v, boundaries, boundaries, capacity, capacity, 0.0, True, False
        )[0]

    def set_lengths(lengths):
        sizes = lengths + [0] * (7 - len(lengths)) + [capacity - sum(lengths)]
        boundaries.copy_(torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()]))
        return sizes

    set_lengths([150, 150, 230, 218, 242, 235])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    results = []
    for lengths in ([150, 150, 230, 218, 242, 235], [3, 128, 192, 256], [256] * 6):
        sizes = set_lengths(lengths)
        graph.replay()
        start = 0
        for length in sizes:
            end = start + length
            if length:
                expected = F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1)[None],
                    k[start:end].transpose(0, 1)[None],
                    v[start:end].transpose(0, 1)[None],
                    is_causal=True,
                    enable_gqa=True,
                )[0].transpose(0, 1)
                actual = output[start:end]
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    ).sqrt()
                )
                assert bool(actual.isfinite().all()) and error < 0.003, error
                results.append(dict(length=length, rrms=error))
            start = end
    elapsed = triton.testing.do_bench_cudagraph(run) * 1000
    args.output.write_text(
        json.dumps(dict(checks=results, last_case_us=elapsed, verdict="PASS"), indent=2) + "\n"
    )
    print(f"Packed attention with changing boundaries and empty sequences: PASS; {elapsed:.2f} us")


if __name__ == "__main__":
    main()
