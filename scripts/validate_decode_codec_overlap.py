"""Replay identical real autoregressive requests with synchronous/queued codec.

Request order and sampling counters are fixed. Compare all generated codes,
model state, codec state and delivered PCM across two generations reusing slots.
This isolates stream safety from service admission and stochastic batch changes.
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
from validate_async_codec import request

from drama3.audio_queue import AudioQueue
from drama3.codec import load_codec, streaming_codec
from drama3.engine import BatchedPrefillGraph, DecodeGraph, Engine, state_pool_size
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fish-source", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = args.corpus.read_bytes()
    samples = [json.loads(s) for s in data.splitlines()][:64]
    model = DramaModel(str(args.checkpoint), slots=state_pool_size(64), max_length=2048)
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    graph = DecodeGraph(model, 64, requests=device)
    first = DecodeGraph(model, 8, first_frame=True, requests=device)
    codec = streaming_codec(load_codec(args.checkpoint, args.fish_source), 64)
    engine = Engine.__new__(Engine)
    engine.device_requests = device
    engine.codec_graphs = codec
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
    prefill = BatchedPrefillGraph(model, 8, triton.next_power_of_2(max(map(len, prompts))))
    for i, prompt in enumerate(prompts):
        history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).cuda()
        device.initialize(i, prompt[-1].cuda(), history, len(prompt) - 1, 500)
    for start in range(0, 64, 8):
        hidden = prefill.run(prompts[start : start + 8], list(range(start, start + 8)))
        first.run(requests[start : start + 8], hidden)
    graph.run(requests)
    model_tensors = [v for pair in (*model.states.values(), *model.kv.values()) for v in pair]
    model_tensors += [
        device.last,
        device.history,
        device.metadata,
        device.codes,
        model.sampling_step,
    ]
    codec_tensors = [*codec.stream.tensors.values(), *codec.stream.kv_tensors.values()]
    initial = [value.clone() for value in model_tensors]
    initial_codec = [value.clone() for value in codec_tensors]
    expected_states, expected_pcm = None, None
    results = []
    for mode in ("sync", "poll", "thread"):
        for value, saved in zip(codec_tensors, initial_codec, strict=True):
            value.copy_(saved)
        events = {}
        engine.audio_queue = (
            None if mode == "sync" else AudioQueue(codec, max_pending=32, threaded=mode == "thread")
        )
        for generation in range(2):
            # Restore model requests while the prior generation's independent
            # codec buffers may still be queued. Codec operations retain FIFO.
            for value, saved in zip(model_tensors, initial, strict=True):
                value.copy_(saved)
            audio_requests = []
            for slot in range(64):
                key = f"{generation}-{slot}"
                events[key] = []
                audio_requests.append(request(slot, events[key].append))
            for step in range(24):
                graph.graph.replay()
                counts = device.metadata[:64, 1].cpu().tolist()
                emit = []
                for req, count in zip(audio_requests, counts, strict=True):
                    req.frame_count = count
                    if count >= req.next_emit or step == 23:
                        emit.append(req)
                engine.emit_audio_batch(emit)
            for req in audio_requests:
                engine.finish_audio(req)
        if engine.audio_queue is not None:
            engine.audio_queue.close()
        torch.cuda.synchronize()
        completion_ok = all(
            values and values[-1] is None and all(isinstance(v, bytes) for v in values[:-1])
            for values in events.values()
        )
        tensors = model_tensors + codec_tensors
        if mode == "sync":
            expected_states = [value.clone() for value in tensors]
            expected_pcm = events
        differences = [
            dict(
                index=i,
                shape=list(value.shape),
                max_abs=float((value.float() - saved.float()).abs().max()),
            )
            for i, (value, saved) in enumerate(zip(tensors, expected_states, strict=True))
            if not torch.equal(value, saved)
        ]
        pcm_mismatches = [key for key in events if events[key] != expected_pcm[key]]
        result = dict(
            mode=mode,
            requests=128,
            steps_per_generation=24,
            completion_ok=completion_ok,
            state_differences=differences,
            pcm_mismatches=pcm_mismatches,
            passed=completion_ok and not differences and not pcm_mismatches,
            corpus_sha256=hashlib.sha256(data).hexdigest(),
            pcm_sha256={
                key: hashlib.sha256(b"".join(values[:-1])).hexdigest()
                for key, values in events.items()
            }
            if completion_ok
            else {},
        )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "pcm_sha256"}), flush=True)
        assert result["passed"], "Codec/decode overlap changed native output, state or delivery"


if __name__ == "__main__":
    main()
