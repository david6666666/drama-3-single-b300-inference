"""Check real prefill/decode overlap with disjoint request states on one GPU.

This excludes first-frame sampling, admission and codec. It measures an upper
bound on a scheduling opportunity, not playable-audio service throughput.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.engine import BatchedPrefillGraph, DecodeGraph, state_pool_size
from drama3.model import DramaModel
from drama3.packed_prefill import PackedPrefillGraph
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=state_pool_size(64), max_length=2048)
    if model.tuned_prefill_moe:
        raise ValueError("This diagnostic requires ordered native prefill reduction")
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    decode = DecodeGraph(model, 64, requests=device)
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][:64]
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
    stream = torch.cuda.Stream()
    capacity = triton.cdiv(sum(map(len, prompts[58:])), 256) * 256
    packed = PackedPrefillGraph(model, 8, capacity, capture_stream=stream)
    span = triton.next_power_of_2(max(map(len, prompts)))
    prefills = {batch: BatchedPrefillGraph(model, batch, span) for batch in (2, 8)}
    first = {
        batch: DecodeGraph(model, batch, first_frame=True, requests=device) for batch in (2, 8)
    }
    requests = [
        SimpleNamespace(slot=i, position=len(prompt) - 1) for i, prompt in enumerate(prompts)
    ]
    for i, prompt in enumerate(prompts):
        history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).cuda()
        device.initialize(i, prompt[-1].cuda(), history, len(prompt) - 1, 500)
    for start in range(0, 58, 8):
        end = min(start + 8, 58)
        batch = end - start
        hidden = prefills[batch].run(prompts[start:end], list(range(start, end)))
        first[batch].run(requests[start:end], hidden)
    packed.run(prompts[58:], list(range(58, 64)))
    decode.run(requests[:58])
    torch.cuda.synchronize()
    tensors = [value for pair in (*model.states.values(), *model.kv.values()) for value in pair]
    tensors += [device.last, device.history, device.metadata, device.codes, model.sampling_step]
    initial = [value.clone() for value in tensors]
    results = []
    expected_state = None
    expected_hidden = None
    for overlap in (False, True):
        for value, saved in zip(tensors, initial, strict=True):
            value.copy_(saved)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        done = torch.cuda.Event()
        begin.record()
        if overlap:
            stream.wait_event(begin)
            with torch.cuda.stream(stream):
                packed.graph.replay()
                done.record()
            for _ in range(5):
                decode.graph.replay()
            torch.cuda.current_stream().wait_event(done)
        else:
            packed.graph.replay()
            for _ in range(5):
                decode.graph.replay()
        end.record()
        end.synchronize()
        hidden = packed.output[:6].clone()
        if not overlap:
            expected_state = [value.clone() for value in tensors]
            expected_hidden = hidden
            result = dict(overlap=False, gpu_ms=begin.elapsed_time(end), passed=True)
        else:
            errors = []
            for index, (value, expected) in enumerate(zip(tensors, expected_state, strict=True)):
                if not torch.equal(value, expected):
                    delta = value.float() - expected.float()
                    errors.append(
                        dict(
                            index=index,
                            shape=list(value.shape),
                            max_abs=float(delta.abs().max()),
                            rrms=float(
                                (
                                    delta.square().mean()
                                    / expected.float().square().mean().clamp_min(1e-12)
                                ).sqrt()
                            ),
                        )
                    )
            same_hidden = torch.equal(hidden, expected_hidden)
            result = dict(
                overlap=True,
                gpu_ms=begin.elapsed_time(end),
                state_differences=errors,
                hidden_exact=same_hidden,
                passed=not errors and same_hidden,
            )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["passed"], "Overlap altered state or output; do not integrate"


if __name__ == "__main__":
    main()
