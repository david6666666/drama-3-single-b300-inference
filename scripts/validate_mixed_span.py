"""Compare mixed graph prompt spans with fixed rows, inputs, RNG and state."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from validate_mixed_layers import Checks, restore, snapshot, state_tensors

from drama3.engine import BatchedPrefillGraph, DecodeGraph, state_pool_size
from drama3.mixed_graph import MixedGraph
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("span", "layout", "conv", "both"), default="span")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    raw = args.corpus.read_bytes()
    checks = Checks(
        args.output,
        dict(
            source_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            source_status=subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True
            ),
            corpus_sha256=hashlib.sha256(raw).hexdigest(),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            environment={k: v for k, v in os.environ.items() if k.startswith("DRAMA3_")},
            hypothesis="Native mixed graph variant; identical rows, inputs, RNG and state.",
            variant=args.variant,
            gate="All outputs and mutable state exact; timing is diagnostic only.",
            stop="Any mismatch rejects the candidate; preserve all cases and raw timings.",
        ),
    )
    try:
        model = DramaModel(args.checkpoint, slots=state_pool_size(64), max_length=2048)
        if model.prefill_moe != "flashinfer" or model.tuned_prefill_moe:
            raise ValueError("Use native FlashInfer prefill MoE without tuning")
        device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
        samples = [json.loads(line) for line in raw.splitlines()][:64]
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
        prefill = BatchedPrefillGraph(model, 8, triton.next_power_of_2(max(map(len, prompts))))
        first = DecodeGraph(model, 8, first_frame=True, requests=device)
        cases = []
        for active_count, arriving_count in [(58, 6), (40, 3), (12, 1)]:
            selected = prompts[active_count : active_count + arriving_count]
            total = sum(map(len, selected))
            capacity = triton.cdiv(total, 256) * 256
            span = triton.cdiv(max([*map(len, selected), capacity - total]), 64) * 64
            d = next(n for n in (16, 32, 48, 64) if n >= active_count)
            flags = {
                "DRAMA3_FUSED_MIXED_LAYOUT": args.variant in ("layout", "both"),
                "DRAMA3_PACKED_CONV_GUARD": args.variant in ("conv", "both"),
            }
            for flag in flags:
                os.environ[flag] = "0"
            control = MixedGraph(
                model, device, d, 8, capacity, max_length=512 if args.variant == "span" else span
            )
            for flag, enabled in flags.items():
                os.environ[flag] = str(int(enabled))
            candidate = MixedGraph(model, device, d, 8, capacity, max_length=span)
            for flag in flags:
                os.environ[flag] = "0"
            cases.append((active_count, arriving_count, selected, control, candidate))
        tensors = state_tensors(model, device)
        for active_count, arriving_count, selected, control, candidate in cases:
            label = f"d{active_count}-p{arriving_count}"
            print(label, flush=True)
            active = [
                SimpleNamespace(slot=i, position=len(prompts[i]) - 1) for i in range(active_count)
            ]
            for i, prompt in enumerate(prompts[:active_count]):
                history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).cuda()
                device.initialize(i, prompt[-1].cuda(), history, len(prompt) - 1, 500)
            for start in range(0, active_count, 8):
                end = min(start + 8, active_count)
                hidden = prefill.run(prompts[start:end], list(range(start, end)))
                first.run(active[start:end], hidden)
            slots = list(range(active_count, active_count + arriving_count))
            for graph in (control, candidate):
                graph.prepare(active, selected, slots)
            initial = snapshot(tensors.values())
            control.graph.replay()
            expected_out = tuple(t.clone() for t in control.output)
            expected_state = snapshot(tensors.values())
            restore(tensors.values(), initial)
            candidate.graph.replay()
            for name, actual, expected in zip(
                ("decode", "prefill"), candidate.output, expected_out, strict=True
            ):
                checks.tensor(f"{label}/{name}", actual, expected, exact=True)
            for (name, actual), expected in zip(tensors.items(), expected_state, strict=True):
                checks.tensor(f"{label}/{name}", actual, expected, exact=True)
            timings = {"control_ms": [], "candidate_ms": []}
            for iteration in range(6):
                arms = [("control_ms", control), ("candidate_ms", candidate)]
                if iteration % 2:
                    arms.reverse()
                for name, graph in arms:
                    restore(tensors.values(), initial)
                    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    graph.graph.replay()
                    end.record()
                    end.synchronize()
                    if iteration:
                        timings[name].append(begin.elapsed_time(end))
            row = dict(
                case=label,
                active=active_count,
                arriving=arriving_count,
                capacity=control.capacity,
                control_span=control.layout.max_length,
                trimmed_span=candidate.layout.max_length,
                prompt_lengths=list(map(len, selected)),
                timings=timings,
                passed=True,
            )
            checks.report["cases"].append(row)
            checks.flush()
            print(json.dumps(row), flush=True)
            del initial, expected_state, expected_out
        checks.report["status"] = "PASS"
        checks.flush()
    except Exception as error:
        checks.report["status"] = "FAIL"
        checks.report["failure"] = dict(type=type(error).__name__, message=str(error))
        checks.flush()
        raise


if __name__ == "__main__":
    main()
