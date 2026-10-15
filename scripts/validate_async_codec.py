"""Check queued PCM parity, slot reuse, bounded draining, cancellation and EOS order."""

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

from drama3.audio_queue import AudioQueue
from drama3.codec import load_codec, streaming_codec
from drama3.engine import Engine
from drama3.runtime_state import DeviceRequests


def request(slot, callback):
    return SimpleNamespace(
        slot=slot,
        frame_count=0,
        emitted=0,
        chunk_index=0,
        next_emit=6,
        pending_audio=0,
        finish_requested=False,
        audio_failed=False,
        cancelled=threading.Event(),
        audio_lock=threading.Lock(),
        emit=callback,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][:4]
    sources = [torch.tensor(s["reference_codes"], device="cuda") for s in samples]
    engine = Engine.__new__(Engine)
    engine.device_requests = DeviceRequests(4, 0, 512)
    engine.codec_graphs = streaming_codec(
        load_codec(args.checkpoint, args.fish_source), 2, dtype=getattr(torch, args.dtype)
    )
    outputs = {}
    for asynchronous in ("sync", "poll", "thread"):
        threaded = asynchronous == "thread"
        events = {}
        engine.audio_queue = (
            AudioQueue(engine.codec_graphs, max_pending=2, threaded=threaded)
            if asynchronous != "sync"
            else None
        )
        for generation in range(2):
            requests = []
            for slot in range(2):
                name = f"{generation}-{slot}"
                events[name] = []
                requests.append(request(slot, events[name].append))
                engine.device_requests.codes[slot, :26].copy_(
                    sources[2 * generation + slot][:, :26].T
                )
            for boundary in (6, 14, 26):
                for req in requests:
                    req.frame_count = boundary
                engine.emit_audio_batch(requests)
            for req in requests:
                engine.finish_audio(req)
            # Reuse model storage before the codec stream is explicitly drained.
            engine.device_requests.codes[:2].zero_()
        if engine.audio_queue is not None:
            engine.audio_queue.close()
        for values in events.values():
            assert values[-1] is None and len(values) == 4, values
            assert all(isinstance(value, bytes) for value in values[:-1])
        outputs[asynchronous] = events
    assert outputs["sync"] == outputs["poll"] == outputs["thread"], (
        "Asynchronous PCM differs or completion arrived too early"
    )
    cancelled = []
    req = request(0, cancelled.append)
    req.cancelled.set()
    req.frame_count = 1
    engine.audio_queue = AudioQueue(engine.codec_graphs, max_pending=2)
    engine.emit_audio_batch([req])
    engine.finish_audio(req)
    engine.audio_queue.close()
    assert not cancelled and req.pending_audio == 0
    engine.audio_queue = AudioQueue(engine.codec_graphs, max_pending=2)
    errors = []
    req = request(0, errors.append)
    req.frame_count = 1
    engine.emit_audio_batch([req])
    engine.finish_audio(req)
    engine.audio_queue.stream.synchronize()
    engine.audio_queue.abort(RuntimeError("injected codec failure"))
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert req.audio_failed and req.pending_audio == 0
    result = dict(
        passed=True,
        codec_dtype=args.dtype,
        checks=[
            "pcm_bytes",
            "slot_reuse",
            "bounded_drain",
            "completion_order",
            "cancelled_delivery",
            "failed_delivery",
        ],
        audio_sha256={
            name: hashlib.sha256(b"".join(values[:-1])).hexdigest()
            for name, values in outputs["thread"].items()
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
