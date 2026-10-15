"""Capture short upstream service traces without changing its source checkout."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import aiohttp
import yaml

from drama3.benchmark import batch, request_body


async def capture(url, corpus, output, concurrency, request_count=None):
    corpus_bytes = corpus.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    samples = [json.loads(row) for row in corpus_bytes.decode().splitlines()]
    bodies = [request_body(sample, "vllm", 500) for sample in samples]
    connector = aiohttp.TCPConnector(limit=64, force_close=True)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        for c in concurrency:
            count = request_count or c
            warm, _ = await batch(session, url + "/v1/audio/speech", samples, bodies, c, c, "warm")
            assert all(row["success"] for row, _ in warm)
            async with session.post(url + "/start_profile", json={}) as response:
                response.raise_for_status()
            started = time.time()
            started_monotonic = time.perf_counter()
            try:
                rows, elapsed = await batch(
                    session, url + "/v1/audio/speech", samples, bodies, c, count, f"profile-c{c}"
                )
            finally:
                async with session.post(url + "/stop_profile", json={}) as response:
                    response.raise_for_status()
                    stopped = await response.text()
            (output / f"c{c}.json").write_text(
                json.dumps(
                    {
                        "diagnostic_only": True,
                        "profile_started_unix": started,
                        "profile_started_monotonic_s": started_monotonic,
                        "profile_request_count": count,
                        "corpus_sha256": corpus_sha256,
                        "concurrency": c,
                        "elapsed_with_profiler_s": elapsed,
                        "stop_profile_response": stopped,
                        "requests": [row for row, _ in rows],
                    },
                    indent=2,
                )
                + "\n"
            )
            assert all(row["success"] for row, _ in rows)
            print(f"Captured C{c}: {len(rows)} successful requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--requests", type=int, help="Profile requests per concurrency; default: C")
    args = parser.parse_args()
    if args.requests is not None and args.requests <= 0:
        parser.error("--requests must be positive")
    if any(c <= 0 for c in args.concurrency):
        parser.error("--concurrency values must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    deploy = yaml.safe_load((args.source / "vllm_omni/deploy/fish_omni.yaml").read_text())
    for stage in deploy["stages"]:
        stage["max_num_seqs"] = 128
        stage["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.output / "traces"),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
        }
    deploy_path = args.output / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    command = [
        str(args.venv / "bin/vllm"),
        "serve",
        str(args.checkpoint),
        "--omni",
        "--served-model-name",
        "fishaudio/drama-3",
        "--port",
        str(args.port),
        "--deploy-config",
        str(deploy_path),
    ]
    metadata = {
        "diagnostic_only": True,
        "command": command,
        "source_sha": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "-C", str(args.source), "status", "--porcelain"], text=True
        ),
        "gpu": args.gpu,
        "profile_request_count": args.requests,
        "profile_request_counts": {str(c): args.requests or c for c in args.concurrency},
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "omp_threads": 8,
    }
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="8",
        HF_MODULES_CACHE=str(args.output / "hf_modules"),
    )
    started = time.monotonic()
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            cwd=args.source,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            url = f"http://127.0.0.1:{args.port}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {process.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() - started > 1800:
                    raise TimeoutError("Readiness deadline exceeded")
                time.sleep(2)
            metadata["startup_to_readiness_s"] = time.monotonic() - started
            (args.output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("Upstream ready", flush=True)
            asyncio.run(capture(url, args.corpus, args.output, args.concurrency, args.requests))
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
