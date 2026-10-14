"""Validate and compare FlashInfer decode backends for Drama 3 geometry."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from drama3.model import AttentionBatch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    result = []
    for backend, tensor_cores in (
        ("auto", False),
        ("auto", True),
        ("cute-dsl", True),
        ("trtllm-gen", True),
    ):
        for size in (1, 8, 64):
            keys = torch.randn(size, 512, 2, 256, device="cuda", dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            q = torch.randn(size, 16, 256, device="cuda", dtype=torch.bfloat16)
            slots = torch.arange(size, device="cuda")
            positions = torch.full((size,), 511, device="cuda")
            row = {"backend": backend, "tensor_cores": tensor_cores, "batch": size}
            print("START", row, flush=True)
            try:
                batch = AttentionBatch(slots, positions, 512, backend, tensor_cores)
                for _ in range(3):
                    batch.run(q, keys, values)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = batch.run(q, keys, values)
                batch.plan(list(range(size)), [300] * size)
                graph.replay()
                expected = F.scaled_dot_product_attention(
                    q[:, :, None],
                    keys[:, :301].transpose(1, 2),
                    values[:, :301].transpose(1, 2),
                    enable_gqa=True,
                ).squeeze(2)
                torch.testing.assert_close(output, expected, atol=0.015625, rtol=0.02)
                row["pass"] = True
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                row["microseconds"] = start.elapsed_time(end) * 10
            except Exception as error:
                row["pass"] = False
                row["error"] = str(error)
            result.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
