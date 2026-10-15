"""Closed-loop streaming PCM benchmark shared by all compared runtimes."""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 2048


def summarize(records: list[dict], elapsed: float) -> dict:
    if not math.isfinite(elapsed) or elapsed <= 0 or not records:
        raise ValueError("A measured run needs records and a positive finite makespan")
    successful = [row for row in records if row["success"]]
    summary = {
        "requests": len(records),
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "makespan_s": elapsed,
        "valid": len(successful) == len(records),
        "throughput_frames_s": None,
        "ttfa_p50_ms": None,
        "ttfa_p95_ms": None,
        "rtf_mean": None,
    }
    if successful:
        # Keep the whole run's makespan, including failed requests and draining.
        summary.update(
            throughput_frames_s=sum(r["audio_samples"] for r in successful)
            / SAMPLES_PER_FRAME
            / elapsed,
            ttfa_p50_ms=float(np.percentile([r["ttfa_s"] for r in successful], 50)) * 1000,
            ttfa_p95_ms=float(np.percentile([r["ttfa_s"] for r in successful], 95)) * 1000,
            rtf_mean=float(np.mean([r["rtf"] for r in successful])),
        )
    return summary


def request_body(sample: dict, backend: str, max_frames: int) -> dict:
    body = {
        "model": "fishaudio/drama-3",
        "input": sample["text"],
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "max_new_tokens": max_frames,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 30,
    }
    if sample.get("reference_audio"):
        reference = "data:audio/wav;base64," + base64.b64encode(
            Path(sample["reference_audio"]).read_bytes()
        ).decode("ascii")
        if backend == "vllm":
            body.update(ref_audio=reference, ref_text=sample["reference_text"])
        else:
            body["references"] = [{"audio": reference, "text": sample["reference_text"]}]
    return body


async def request(
    session: aiohttp.ClientSession,
    endpoint: str,
    body: dict,
    request_id: str,
    sample_id: str,
) -> tuple[dict, bytes]:
    record = {"request_id": request_id, "sample_id": sample_id, "success": False}
    chunks = []
    chunk_times = []
    start = time.perf_counter()
    record["dispatch_s"] = start
    try:
        async with session.post(endpoint, json=body) as response:
            record["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {(await response.text())[:1000]}")
            content_type = response.headers.get("content-type", "").split(";")[0]
            if content_type != "audio/pcm":
                raise ValueError(f"Expected audio/pcm, got {content_type!r}")
            sample_rate = int(response.headers.get("x-sample-rate", SAMPLE_RATE))
            channels = int(response.headers.get("x-channels", 1))
            bits = int(response.headers.get("x-bit-depth", 16))
            record["audio_headers"] = {
                name: value
                for name, value in response.headers.items()
                if name.lower().startswith("x-")
            }
            if (sample_rate, channels, bits) != (SAMPLE_RATE, 1, 16):
                raise ValueError(f"Unexpected PCM format: {(sample_rate, channels, bits)}")
            async for chunk in response.content.iter_any():
                if chunk:
                    chunk_times.append(time.perf_counter())
                    chunks.append(chunk)
        end = time.perf_counter()
        audio = b"".join(chunks)
        if not audio or len(audio) % 2:
            raise ValueError(f"Empty or unaligned PCM: {len(audio)} bytes")
        samples = len(audio) // 2
        duration = samples / SAMPLE_RATE
        pcm = np.frombuffer(audio, dtype="<i2")
        record.update(
            success=True,
            ttfa_s=chunk_times[0] - start,
            last_audio_s=chunk_times[-1] - start,
            completion_s=end - start,
            audio_samples=samples,
            acoustic_frames=samples / SAMPLES_PER_FRAME,
            audio_duration_s=duration,
            rtf=(chunk_times[-1] - start) / duration,
            transport_chunks=len(chunks),
            chunk_arrivals_s=[timestamp - start for timestamp in chunk_times],
            chunk_bytes=[len(chunk) for chunk in chunks],
            audio_sha256=hashlib.sha256(audio).hexdigest(),
            rms=float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) / 32768),
            peak=float(np.max(np.abs(pcm.astype(np.int32))) / 32768),
            finish_reason="unavailable_in_raw_pcm",
        )
        return record, audio
    except (TimeoutError, aiohttp.ClientError, ValueError) as error:
        record.update(error=str(error), completion_s=time.perf_counter() - start)
        return record, b""


async def batch(
    session: aiohttp.ClientSession,
    endpoint: str,
    samples: list[dict],
    bodies: list[dict],
    concurrency: int,
    requests: int,
    label: str,
) -> tuple[list[tuple[dict, bytes]], float]:
    indices = iter(range(requests))
    outputs = []

    async def worker() -> None:
        for index in indices:
            sample_index = index % len(samples)
            outputs.append(
                await request(
                    session,
                    endpoint,
                    bodies[sample_index],
                    f"{label}-{index:05d}",
                    samples[sample_index]["id"],
                )
            )

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return outputs, time.perf_counter() - start


async def run(args: argparse.Namespace) -> None:
    corpus_bytes = args.corpus.read_bytes()
    samples = [json.loads(line) for line in corpus_bytes.splitlines() if line.strip()]
    if not samples:
        raise ValueError("The corpus must not be empty")
    bodies = [request_body(sample, args.backend, args.max_frames) for sample in samples]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "frame_measure": "returned_waveform_samples / 2048; excludes EOS",
        "rtf_boundary": "dispatch_to_last_audio",
        "finish_reason_coverage": False,
        "connection_policy": "fresh TCP connection per request; no automatic retries",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    # Runs save WAV files between measured batches. A pooled socket can expire
    # at the server while its close notification waits on this event loop.
    # Use the same fresh-connection policy for every backend and keep failures.
    connector = aiohttp.TCPConnector(limit=max(args.concurrency), force_close=True)
    endpoint = args.url.rstrip("/") + "/v1/audio/speech"
    summaries = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(args.url.rstrip("/") + "/health") as response:
            response.raise_for_status()
        if args.warm_references:
            reference_warmup, preparation_elapsed = await batch(
                session, endpoint, samples, bodies, 1, len(samples), "reference-warmup"
            )
            (args.output / "reference-warmup.json").write_text(
                json.dumps(
                    {
                        "elapsed_s": preparation_elapsed,
                        "requests": [row for row, _ in reference_warmup],
                    },
                    indent=2,
                )
                + "\n"
            )
            if any(not row["success"] for row, _ in reference_warmup):
                raise RuntimeError("Reference warmup failed")
        for concurrency in args.concurrency:
            warmup, _ = await batch(
                session,
                endpoint,
                samples,
                bodies,
                concurrency,
                max(args.warmup, concurrency),
                f"c{concurrency}-warmup",
            )
            (args.output / f"c{concurrency}-warmup.json").write_text(
                json.dumps([row for row, _ in warmup], indent=2) + "\n"
            )
            if any(not row["success"] for row, _ in warmup):
                raise RuntimeError("Warmup failed; inspect the saved request records")
            for repeat in range(args.repeats):
                label = f"c{concurrency}-r{repeat}"
                outputs, elapsed = await batch(
                    session, endpoint, samples, bodies, concurrency, args.requests, label
                )
                records = [row for row, _ in outputs]
                summary = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **summarize(records, elapsed),
                }
                summaries.append(summary)
                (args.output / f"{label}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in records)
                )
                (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                if args.save_audio:
                    for row, audio in outputs:
                        if not row["success"]:
                            continue
                        with wave.open(str(args.output / f"{row['request_id']}.wav"), "wb") as wav:
                            wav.setnchannels(1)
                            wav.setsampwidth(2)
                            wav.setframerate(SAMPLE_RATE)
                            wav.writeframes(audio)
                print(json.dumps(summary), flush=True)
                if not summary["valid"]:
                    raise RuntimeError("Measured requests failed; run is ineligible for promotion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--backend", choices=["sglang", "vllm", "standalone"], required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--requests", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--warm-references", action="store_true")
    args = parser.parse_args()
    if min(*args.concurrency, args.requests, args.repeats, args.max_frames, args.timeout) <= 0:
        parser.error("Concurrency, requests, repeats, frame limit and timeout must be positive")
    if args.requests < max(args.concurrency):
        parser.error("Request count must cover the largest concurrency")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
