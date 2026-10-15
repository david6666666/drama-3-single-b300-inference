"""Compare native MoE scheduling/fusion graphs from identical real decode state.

All eight routed experts, quantization boundaries and shared expert operators
are retained. This is a state/parity and graph timing diagnostic, not serving.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.engine import BatchedPrefillGraph, DecodeGraph, state_pool_size
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(2, 4, 8, 16, 32, 64), required=True)
    parser.add_argument(
        "--optimization", choices=("shared-expert", "moe-input"), default="shared-expert"
    )
    args = parser.parse_args()
    data = args.corpus.read_bytes()
    samples = [json.loads(s) for s in data.splitlines()][: args.batch]
    model = DramaModel(args.checkpoint, slots=state_pool_size(args.batch), max_length=2048)
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    model.moe_shared_stream = None
    model.fused_moe_input = False
    control = DecodeGraph(model, args.batch, requests=device)
    shared_stream = torch.cuda.Stream()
    if args.optimization == "shared-expert":
        model.moe_shared_stream = shared_stream
    else:
        if args.batch <= 8:
            parser.error("Fused MoE input applies only to the native FlashInfer batch path")
        model.fused_moe_input = True
    candidate = DecodeGraph(model, args.batch, requests=device)
    # Prefill and first-frame controls use the original serial computation.
    model.moe_shared_stream = None
    model.fused_moe_input = False
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
    requests = [SimpleNamespace(slot=i, position=len(p) - 1) for i, p in enumerate(prompts)]
    batch = min(8, args.batch)
    span = triton.next_power_of_2(max(map(len, prompts)))
    prefill = BatchedPrefillGraph(model, batch, span)
    first = DecodeGraph(model, batch, first_frame=True, requests=device)
    for i, prompt in enumerate(prompts):
        history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).cuda()
        device.initialize(i, prompt[-1].cuda(), history, len(prompt) - 1, 500)
    for start in range(0, args.batch, batch):
        end = start + batch
        hidden = prefill.run(prompts[start:end], list(range(start, end)))
        first.run(requests[start:end], hidden)
    # Populate both graph input buffers, then save one common starting state.
    control.run(requests)
    candidate.run(requests)
    tensors = [value for pair in (*model.states.values(), *model.kv.values()) for value in pair]
    tensors += [device.last, device.history, device.metadata, device.codes, model.sampling_step]
    initial = [value.clone() for value in tensors]
    results = []
    expected = None
    expected_frames = None
    for overlap, graph in ((False, control), (True, candidate)):
        for value, saved in zip(tensors, initial, strict=True):
            value.copy_(saved)
        frames = []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.graph.replay()
            frames.append(graph.output.clone())
        end.record()
        end.synchronize()
        if not overlap:
            expected = [value.clone() for value in tensors]
            expected_frames = torch.stack(frames)
            differences = []
            exact_frames = True
        else:
            differences = [
                dict(
                    index=i,
                    shape=list(value.shape),
                    max_abs=float((value.float() - saved.float()).abs().max()),
                )
                for i, (value, saved) in enumerate(zip(tensors, expected, strict=True))
                if not torch.equal(value, saved)
            ]
            exact_frames = torch.equal(torch.stack(frames), expected_frames)
        row = dict(
            optimization=args.optimization,
            overlap=overlap,
            batch=args.batch,
            gpu_ms_per_step=begin.elapsed_time(end) / 10,
            frames_exact=exact_frames,
            state_differences=differences,
            passed=exact_frames and not differences,
            corpus_sha256=hashlib.sha256(data).hexdigest(),
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert row["passed"], "Native stream scheduling changed output/state; stop"


if __name__ == "__main__":
    main()
