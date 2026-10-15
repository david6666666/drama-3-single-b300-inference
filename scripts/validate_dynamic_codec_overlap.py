"""Replay dynamic real requests with identical admission, RNG and codec inputs.

This is a correctness diagnostic, not a service/quality or throughput benchmark.
Admission occurs at scripted decode steps; native EOS retires requests. The
serial trace must match both overlapping modes exactly, including slot reuse,
prefill/decode graph selection, generated codes, PCM and final mutable state.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from validate_async_codec import request

from drama3.audio_queue import AudioQueue
from drama3.codec import load_codec, streaming_codec
from drama3.engine import DecodeGraph, Engine, state_pool_size
from drama3.model import DramaModel
from drama3.packed_prefill import PackedPrefillGraph
from drama3.runtime_state import DeviceRequests

WORKSPACE_OBSERVER = r"""
#include <ATen/cuda/CUDAContextLight.h>
#include <array>
#include <shared_mutex>
#include <vector>
std::vector<std::array<uint64_t, 5>> snapshot() {
  std::vector<std::array<uint64_t, 5>> rows;
  auto append = [&](uint64_t kind, at::cuda::WorkspaceMapWithMutex& ws) {
    std::shared_lock<std::shared_mutex> lock(ws.mutex);
    for (const auto& entry : ws.map) {
      rows.push_back({kind,
        reinterpret_cast<uint64_t>(std::get<0>(entry.first)),
        reinterpret_cast<uint64_t>(std::get<1>(entry.first)),
        reinterpret_cast<uint64_t>(entry.second.first.get()),
        static_cast<uint64_t>(entry.second.second)});
    }
  };
  append(0, at::cuda::cublas_handle_stream_to_workspace());
  append(1, at::cuda::cublaslt_handle_stream_to_workspace());
  return rows;
}
"""


def workspace_observer():
    """Read the installed ATen workspace maps; never allocate/clear their buffers."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="drama3_workspace_observer",
        cpp_sources=WORKSPACE_OBSERVER,
        functions=["snapshot"],
        with_cuda=True,
        extra_cflags=["-O0"],
    )


def inspect_workspaces(observer, model_stream, codec_stream, dedicated):
    rows = [
        dict(kind=("cublas", "cublaslt")[kind], handle=handle, stream=stream, ptr=ptr, size=size)
        for kind, handle, stream, ptr, size in observer.snapshot()
    ]
    model = [row for row in rows if row["stream"] == model_stream]
    codec = [row for row in rows if row["stream"] == codec_stream]
    overlaps = [
        [left, right]
        for left in model
        for right in codec
        if left["ptr"] < right["ptr"] + right["size"] and right["ptr"] < left["ptr"] + left["size"]
    ]
    result = dict(
        scope="Installed ATen cuBLAS/cuBLASLt maps; does not inspect other libraries' scratch",
        model_capture_stream=model_stream,
        codec_capture_stream=codec_stream,
        model=model,
        codec=codec,
        overlapping_ranges=overlaps,
    )
    if dedicated:
        assert model_stream != codec_stream, result
        assert model and codec, "Capture streams must have observable ATen BLAS workspaces"
        assert not overlaps, result
    return result


class ObservedCodec:
    def __init__(self, graphs):
        self.graphs = graphs
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.graphs, name)

    def __call__(self, codes, *, rows):
        self.calls.append((len(codes), codes.shape[-1]))
        return self.graphs(codes, rows=rows)


def sha(value):
    return hashlib.sha256(value).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fish-source", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", choices=("default", "dedicated"), default="dedicated")
    parser.add_argument("--build-observer-only", action="store_true")
    args = parser.parse_args()
    observer = workspace_observer()
    if args.build_observer_only:
        print(json.dumps(dict(observer_built=True, workspaces=observer.snapshot())), flush=True)
        return
    if not all((args.checkpoint, args.fish_source, args.corpus, args.output)):
        parser.error("checkpoint, fish-source, corpus and output are required for GPU validation")
    os.environ["DRAMA3_CODEC_DEDICATED_CAPTURE_STREAM"] = str(int(args.capture == "dedicated"))
    repo = Path(__file__).resolve().parents[1]
    data = args.corpus.read_bytes()
    result = dict(
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ),
        corpus_sha256=sha(data),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        capture=args.capture,
        environment={k: v for k, v in os.environ.items() if k.startswith("DRAMA3_")},
        hypothesis="A private codec capture stream isolates ATen BLAS scratch from model graphs.",
        controls="Fixed graphs, initial model/codec state, RNG, corpus and scripted arrivals.",
        gate="Exact roster/codes, PCM/completion and final mutable state; no E2E claim.",
        stop="Any divergence fails; dedicated capture must have disjoint observed BLAS ranges.",
        modes=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    model = DramaModel(str(args.checkpoint), slots=state_pool_size(64), max_length=2048)
    if model.prefill_moe != "flashinfer" or model.tuned_prefill_moe:
        raise ValueError("Use native FlashInfer prefill MoE without tuning")
    device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
    samples = [json.loads(line) for line in data.splitlines()][:80]
    prompts = [
        build_drama3_prompt(
            sample["text"],
            model.weights.tokenizer,
            model.config,
            [sample["reference_text"]],
            [torch.tensor(sample["reference_codes"])],
        )
        for sample in samples
    ]
    # Fixed arrival groups exercise partially populated prefill graphs and
    # different token capacities. The final group reuses slots 0..15 immediately
    # after their old requests' tail chunks have been submitted.
    phases = [
        ([(0, 8)], [], 8),
        ([(8, 8)], [], 8),
        ([(16, 3), (19, 5), (24, 8)], [], 8),
        ([(32, 1), (33, 7), (40, 8)], [], 8),
        ([(48, 4), (52, 8), (60, 4)], [], 24),
        ([(0, 8), (8, 8)], list(range(16)), 128),
    ]
    arrivals, cursor = [], 0
    for groups, _, _ in phases:
        for slot, count in groups:
            selected = prompts[cursor : cursor + count]
            capacity = triton.cdiv(sum(map(len, selected)), 256) * 256
            arrivals.append((slot, count, cursor, capacity))
            cursor += count
    prefill = {
        capacity: PackedPrefillGraph(model, 8, capacity)
        for capacity in sorted({row[3] for row in arrivals})
    }
    decode = {
        size: DecodeGraph(model, size, requests=device) for size in (1, 2, 4, 8, 16, 32, 48, 64)
    }
    first = DecodeGraph(model, 8, first_frame=True, requests=device)
    model_stream = torch.cuda.graph.default_capture_stream.cuda_stream
    codec = ObservedCodec(streaming_codec(load_codec(args.checkpoint, args.fish_source), 64))
    codec_stream = (
        codec.capture_stream.cuda_stream if codec.capture_stream is not None else model_stream
    )
    result["workspace"] = inspect_workspaces(
        observer, model_stream, codec_stream, args.capture == "dedicated"
    )
    save()
    engine = Engine.__new__(Engine)
    engine.device_requests, engine.codec_graphs = device, codec
    tensors = [v for pair in (*model.states.values(), *model.kv.values()) for v in pair]
    tensors += [device.last, device.history, device.metadata, device.codes, model.sampling_step]
    tensors += [*codec.stream.tensors.values(), *codec.stream.kv_tensors.values()]
    initial = [value.clone() for value in tensors]
    rng = torch.cuda.get_rng_state()
    expected_trace = expected_pcm = expected_state = None
    for mode in ("sync", "poll", "thread"):
        for value, saved in zip(tensors, initial, strict=True):
            value.copy_(saved)
        torch.cuda.set_rng_state(rng)
        codec.calls.clear()
        events, active, trace, frames = {}, {}, [], {}
        engine.audio_queue = (
            None if mode == "sync" else AudioQueue(codec, max_pending=32, threaded=mode == "thread")
        )

        def record(entry, expected=expected_trace, trace=trace, mode=mode):
            if expected is not None:
                index = len(trace)
                assert index < len(expected) and entry == expected[index], (
                    mode,
                    index,
                    entry,
                    expected[index] if index < len(expected) else None,
                )
            trace.append(entry)

        def retire(slots, active=active, frames=frames, record=record):
            retiring = [active[slot] for slot in slots if slot in active]
            engine.emit_audio_batch(retiring)
            for req in retiring:
                engine.finish_audio(req)
                frames[req.key] = req.frame_count
                del active[req.slot]
            record(dict(retire=[req.key for req in retiring]))

        def update(reqs, outputs, operation, record=record, retire=retire):
            host = outputs.cpu()
            counts = device.metadata[[req.slot for req in reqs], 1:4].cpu().tolist()
            for req, (count, _, _) in zip(reqs, counts, strict=True):
                req.frame_count = count
            record(
                dict(
                    operation=operation,
                    roster=[req.key for req in reqs],
                    codes_sha256=sha(host.numpy().tobytes()),
                    metadata=counts,
                )
            )
            engine.emit_audio_batch([req for req in reqs if req.frame_count >= req.next_emit])
            retire([req.slot for req, (_, _, done) in zip(reqs, counts, strict=True) if done])

        arrival_index = 0
        try:
            for groups, retired, steps in phases:
                retire(retired)
                for _group in groups:
                    slot, count, start, capacity = arrivals[arrival_index]
                    arrival_index += 1
                    new = []
                    for offset, prompt in enumerate(prompts[start : start + count]):
                        key = str(start + offset)
                        events[key] = []
                        req = request(slot + offset, events[key].append)
                        req.key, req.position = key, len(prompt) - 1
                        assert req.slot not in active
                        active[req.slot] = req
                        new.append(req)
                        history = F.pad(
                            prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1
                        ).cuda()
                        device.initialize(req.slot, prompt[-1].cuda(), history, req.position, 500)
                    record(
                        dict(
                            admit=[req.key for req in new],
                            slots=[req.slot for req in new],
                            prefill_capacity=capacity,
                        )
                    )
                    hidden = prefill[capacity].run(
                        prompts[start : start + count], [req.slot for req in new]
                    )
                    output = first.run(new, hidden)
                    update(new, output, "first_frame")
                for _ in range(steps):
                    if not active:
                        record(dict(empty=True))
                        continue
                    reqs = list(active.values())
                    size = next(size for size in decode if size >= len(reqs))
                    update(reqs, decode[size].run(reqs), f"decode_{size}")
            retire(list(active))
        finally:
            if engine.audio_queue is not None:
                engine.audio_queue.close()
        torch.cuda.synchronize()
        completion_ok = all(
            values and values[-1] is None and all(isinstance(v, bytes) for v in values[:-1])
            for values in events.values()
        )
        if mode == "sync":
            expected_trace, expected_pcm = trace, events
            expected_state = [value.clone() for value in tensors]
        differences = [
            dict(
                index=i,
                shape=list(value.shape),
                max_abs=float((value.float() - saved.float()).abs().max()),
            )
            for i, (value, saved) in enumerate(zip(tensors, expected_state, strict=True))
            if not torch.equal(value, saved)
        ]
        pcm_mismatches = [key for key in events if events[key] != expected_pcm[key]]
        check = dict(
            mode=mode,
            completion_ok=completion_ok,
            state_differences=differences,
            pcm_mismatches=pcm_mismatches,
            request_frames=frames,
            trace_sha256=sha(json.dumps(trace, sort_keys=True).encode()),
            trace_length=len(trace),
            trace_matches=trace == expected_trace,
            codec_shapes=sorted(set(codec.calls)),
            codec_calls=len(codec.calls),
            decode_shapes=sorted({e["operation"] for e in trace if "operation" in e}),
            prefill_capacities=sorted(prefill),
            pcm_sha256={key: sha(b"".join(values[:-1])) for key, values in events.items()}
            if completion_ok
            else {},
        )
        check["passed"] = (
            completion_ok and not differences and not pcm_mismatches and trace == expected_trace
        )
        result["modes"].append(check)
        save()
        print(
            json.dumps(
                {k: v for k, v in check.items() if k not in ("request_frames", "pcm_sha256")}
            ),
            flush=True,
        )
        assert check["passed"], "Dynamic overlap changed native outputs, state or delivery"
    result["passed"] = all(check["passed"] for check in result["modes"])
    result["trace"] = expected_trace
    save()


if __name__ == "__main__":
    main()
