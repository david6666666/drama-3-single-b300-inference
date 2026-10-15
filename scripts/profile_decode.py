"""Profile captured standalone decode, excluding server and codec overhead."""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from drama3.engine import DecodeGraph
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cold-gemv", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cold_gemv:
        import triton

        import drama3.kernels as kernels

        def cold_gemv(x, weight, rows=4):
            n, k = weight.shape
            rows = 2 if n == 4096 and k == 2048 else 1
            warps = 4 if n >= 9216 or k == 512 else 8
            out = torch.empty((1, n), dtype=x.dtype, device=x.device)
            kernels.gemv_kernel[(triton.cdiv(n, rows),)](
                x, weight, out, n, k, rows, triton.next_power_of_2(k), num_warps=warps
            )
            return out

        kernels.gemv = cold_gemv
    model = DramaModel(args.checkpoint, slots=2 * args.batch, max_length=args.max_length)
    graph = DecodeGraph(model, args.batch)
    graph.positions.fill_(300)
    graph.batch.plan(graph.slots.cpu().tolist(), [300] * args.batch)
    for _ in range(10):
        graph.graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(50):
        graph.graph.replay()
    end.record()
    torch.cuda.synchronize()
    result = {
        "batch": args.batch,
        "max_length": args.max_length,
        "gdn_state_precision": model.state_precision,
        "dense_precision": model.dense_precision,
        "moe_layout": model.moe_layout,
        "prefill_moe": model.prefill_moe,
        "decode_ms": start.elapsed_time(end) / 50,
    }
    (args.output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            graph.graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output / "trace.json.gz"))
    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=50)
    (args.output / "table.txt").write_text(table + "\n")
    print(table, flush=True)


if __name__ == "__main__":
    main()
