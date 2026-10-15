"""Check native GEMM correctness when captured graphs replay concurrently.

PyTorch's BLAS workspace cache is keyed by handle and capture stream. Compare
shared/default capture with explicit distinct capture streams before expanding
model/codec overlap. This is a diagnostic, not an asserted race in the model.
"""

import argparse
import json
from pathlib import Path

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for separate in (False, True):
        graphs, outputs, expected, replay_streams, keepalive = [], [], [], [], []
        for _branch in range(2):
            stream = torch.cuda.Stream()
            replay_streams.append(stream)
            inputs = [
                (
                    torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                    torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
                )
                for m, n, k in ((4, 12352, 2048), (64, 2048, 2048), (16, 4096, 4096))
            ]
            keepalive.append(inputs)

            def forward(inputs=inputs):
                return [
                    torch.mm(x, weight.T, out_dtype=torch.float32)
                    for _ in range(8)
                    for x, weight in inputs
                ]

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream if separate else None):
                values = forward()
            graph.replay()
            expected.append([value.clone() for value in values])
            graphs.append(graph)
            outputs.append(values)
        torch.cuda.synchronize()
        for stream, graph in zip(replay_streams, graphs, strict=True):
            with torch.cuda.stream(stream):
                for _ in range(100):
                    graph.replay()
        torch.cuda.synchronize()
        errors = []
        for actual_group, expected_group in zip(outputs, expected, strict=True):
            for actual, reference in zip(actual_group, expected_group, strict=True):
                error = float(
                    (
                        (actual - reference).square().mean()
                        / reference.square().mean().clamp_min(1e-12)
                    ).sqrt()
                )
                errors.append(
                    dict(
                        rrms=error,
                        exact=torch.equal(actual, reference),
                        finite=bool(actual.isfinite().all()),
                    )
                )
        result = dict(
            separate_capture_streams=separate,
            comparisons=errors,
            passed=all(row["exact"] and row["finite"] for row in errors),
            torch_git=torch.version.git_version,
        )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    main()
