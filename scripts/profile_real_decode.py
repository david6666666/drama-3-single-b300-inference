"""Profile heterogeneous voice-clone decode after real prompt prefill."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from torch.profiler import ProfilerActivity, profile

from drama3.engine import BatchedPrefillGraph, DecodeGraph, state_pool_size
from drama3.kernels import linear
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mxfp4-diagnostic", action="store_true")
    parser.add_argument("--mxfp8-diagnostic", action="store_true")
    parser.add_argument("--small-moe-diagnostic", action="store_true")
    parser.add_argument("--small-moe-backends", action="store_true")
    parser.add_argument("--routing-distribution", action="store_true")
    parser.add_argument("--projection-diagnostic", action="store_true")
    parser.add_argument("--small-moe-tiles", choices=("default", "hot", "cold"), default="default")
    args = parser.parse_args()
    if args.batch not in (1, 2, 4) and (args.batch < 8 or args.batch % 8):
        parser.error("Batch must be 1, 2, 4 or a positive multiple of eight")
    prefill_batch = min(args.batch, 8)
    if args.small_moe_tiles != "default":
        import drama3.model as model_module

        model_module.small_batch_moe = partial(
            model_module.small_batch_moe,
            up_rows=8 if args.small_moe_tiles == "hot" else 2,
            down_rows=8,
            down_warps=4,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.corpus.read_bytes()
    samples = [json.loads(s) for s in data.splitlines()][: args.batch]
    model = DramaModel(args.checkpoint, slots=state_pool_size(args.batch), max_length=2048)
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
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    requests = [SimpleNamespace(slot=i, position=len(p) - 1) for i, p in enumerate(prompts)]
    original = model.moe
    indices = {id(layer.mlp): i for i, layer in enumerate(model.backbone.layers)}
    activations = {}
    projections = {}
    if args.projection_diagnostic:
        import drama3.model as model_module

        def capture_projection(x, weight):
            if torch.cuda.is_current_stream_capturing():
                projections[id(weight)] = (x, weight)
            return linear(x, weight)

        model_module.linear = capture_projection

    def capture(x, layer, decode=False):
        if decode and (args.routing_distribution or indices[id(layer)] in (0, 19, 39)):
            activations[indices[id(layer)]] = x
        return original(x, layer, decode)

    model.moe = capture
    graph = DecodeGraph(model, args.batch, requests=device)
    model.moe = original
    if args.projection_diagnostic:
        model_module.linear = linear
    first = DecodeGraph(model, prefill_batch, first_frame=True, requests=device)
    span = max(128, triton.next_power_of_2(max(map(len, prompts))))
    prefill = BatchedPrefillGraph(model, prefill_batch, span)
    for i, prompt in enumerate(prompts):
        history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).to("cuda")
        device.initialize(i, prompt[-1].to("cuda"), history, len(prompt) - 1, 500)
    for start in range(0, args.batch, prefill_batch):
        end = start + prefill_batch
        hidden = prefill.run(prompts[start:end], list(range(start, end)))
        first.run(requests[start:end], hidden)
    graph.run(requests)
    for _ in range(7):
        graph.graph.replay()
    torch.cuda.synchronize()
    initial_counts = device.metadata[: args.batch, 1].cpu().tolist()
    initial_done = int(device.metadata[: args.batch, 3].sum())
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(10):
        graph.graph.replay()
    end.record()
    end.synchronize()
    result = dict(
        batch=args.batch,
        corpus_sha256=hashlib.sha256(data).hexdigest(),
        prompt_lengths=list(map(len, prompts)),
        initial_counts=initial_counts,
        initial_done=initial_done,
        final_done=int(device.metadata[: args.batch, 3].sum()),
        decode_ms=begin.elapsed_time(end) / 10,
        state_dtype=model.state_precision,
        moe_layout=model.moe_layout,
        packed_projection=model.packed_moe_projection,
        small_moe_tiles=args.small_moe_tiles,
        description="Ten actual autoregressive steps; completed slots remain in graph",
    )
    (args.output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            graph.graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output / "trace.json.gz"))
    (args.output / "table.txt").write_text(
        prof.key_averages().table(sort_by="self_device_time_total", row_limit=40) + "\n"
    )
    rows = []
    routing_rows = []
    for index, x in activations.items():
        layer = model.backbone.layers[index].mlp
        router = (
            linear(x, layer.packed_input_projection)[:, 1024:1280]
            if model.packed_moe_projection
            else linear(x, layer.gate.weight)
        )
        ids = router.float().topk(8, dim=-1).indices
        counts = torch.bincount(ids.flatten(), minlength=256)
        row = dict(
            layer=index,
            unique_experts=int((counts > 0).sum()),
            max_expert_tokens=int(counts.max()),
            moe_us=triton.testing.do_bench_cudagraph(partial(original, x, layer, True)) * 1000,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.routing_distribution:
            probabilities = router.float().softmax(-1)
            weights = probabilities.gather(1, ids)
            normalized = weights / weights.sum(-1, keepdim=True)
            cumulative = normalized.cumsum(-1)
            routing_rows.append(
                dict(
                    layer=index,
                    unique_experts=row["unique_experts"],
                    top8_total_probability_mean=float(weights.sum(-1).mean()),
                    retained_mass_mean={
                        str(k): float(cumulative[:, k - 1].mean()) for k in (1, 2, 4, 6, 8)
                    },
                    experts_for_99pct_mean=float(((cumulative < 0.99).sum(-1) + 1).float().mean()),
                    ids=ids.cpu().tolist(),
                    normalized_weights=normalized.cpu().tolist(),
                )
            )
    (args.output / "moe.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.routing_distribution:
        (args.output / "routing.json").write_text(json.dumps(routing_rows, indent=2) + "\n")
    if args.projection_diagnostic:
        from diagnostic_bf16_projections import benchmark

        benchmark(projections, args.output / "projections.json")
    if args.small_moe_backends:
        rows = []
        original_threshold = model.small_moe_max_batch
        for index, x in activations.items():
            layer = model.backbone.layers[index].mlp
            model.small_moe_max_batch = 8
            expected = original(x, layer, True)
            for name, threshold in (("gemv", 8), ("flashinfer", 0)):
                model.small_moe_max_batch = threshold
                fn = partial(original, x, layer, True)
                actual = fn()
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    ).sqrt()
                )
                passed = bool(actual.isfinite().all()) and error < 0.03
                row = dict(layer=index, backend=name, rrms=error, passed=passed)
                if passed:
                    row["us"] = triton.testing.do_bench_cudagraph(fn) * 1000
                rows.append(row)
                (args.output / "small-moe-backends.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                print(json.dumps(row), flush=True)
        model.small_moe_max_batch = original_threshold
    if args.mxfp4_diagnostic:
        from diagnostic_mxfp4_moe import benchmark

        rows = []
        for index, x in activations.items():
            row = dict(layer=index, **benchmark(model, model.backbone.layers[index].mlp, x))
            rows.append(row)
            (args.output / "mxfp4.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if args.mxfp8_diagnostic:
        from diagnostic_mxfp8_moe import benchmark

        rows = []
        for index, x in activations.items():
            row = dict(layer=index, **benchmark(model, model.backbone.layers[index].mlp, x))
            rows.append(row)
            (args.output / "mxfp8.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if not row["passed"]:
                break
    if args.small_moe_diagnostic:
        from diagnostic_small_moe import benchmark

        rows = []
        for index, x in activations.items():
            row = dict(layer=index, **benchmark(model, model.backbone.layers[index].mlp, x))
            rows.append(row)
            (args.output / "small-moe.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
