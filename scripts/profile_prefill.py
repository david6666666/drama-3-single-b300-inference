"""Profile real voice-clone prefills; random MoE routing is not representative."""

import argparse
import json
from pathlib import Path

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from torch.profiler import ProfilerActivity, profile

from drama3.engine import BatchedPrefillGraph
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = DramaModel(args.checkpoint, slots=4 * args.batch, max_length=512)
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][: args.batch]
    prompts = [
        build_drama3_prompt(
            s["text"],
            model.weights.tokenizer,
            model.config,
            [s["reference_text"]],
            [torch.tensor(s["reference_codes"])],
        )
        for s in samples
    ]
    span = max(128, triton.next_power_of_2(max(map(len, prompts))))
    graph = BatchedPrefillGraph(model, args.batch, span)
    graph.run(prompts, list(range(args.batch)))
    for _ in range(3):
        graph.graph.replay()
    # A captured graph cannot be replayed from inside another stream capture.
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(20):
        graph.graph.replay()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 20
    result = dict(
        batch=args.batch,
        span=span,
        prompt_lengths=list(map(len, prompts)),
        prefill_ms=elapsed,
        backend=model.prefill_moe,
        layout=model.moe_layout,
        state_precision=model.state_precision,
        packed_projection=model.packed_moe_projection,
        prefill_conv=model.prefill_conv_backend,
        tuned_prefill_moe=model.tuned_prefill_moe,
    )
    (args.output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            graph.graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output / "trace.json.gz"))
    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
    (args.output / "table.txt").write_text(table + "\n")
    print(table, flush=True)


if __name__ == "__main__":
    main()
