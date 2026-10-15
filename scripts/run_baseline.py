"""Start, benchmark, and clean up one pinned single-GPU upstream baseline."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sglang", "vllm", "standalone"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--bench-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--requests", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="8",
        HF_MODULES_CACHE=str(args.output / "hf_modules"),
    )
    if args.backend == "standalone":
        if args.fish_source is None:
            parser.error("--fish-source is required for the standalone runtime")
        environment["PYTHONPATH"] = str(args.source / "src")
        command = [
            str(args.venv / "bin/python"),
            "-m",
            "drama3.server",
            "--checkpoint",
            str(args.checkpoint),
            "--fish-source",
            str(args.fish_source),
            "--slots",
            str(args.capacity),
            "--port",
            str(args.port),
        ]
    elif args.backend == "sglang":
        command = [
            str(args.venv / "bin/sgl-omni"),
            "serve",
            "--model-path",
            str(args.checkpoint),
            "--port",
            str(args.port),
            "--tts_engine.factory.max_running_requests",
            str(args.capacity),
        ]
    else:
        import yaml

        deploy = yaml.safe_load((args.source / "vllm_omni/deploy/fish_omni.yaml").read_text())
        for stage in deploy["stages"]:
            stage["max_num_seqs"] = args.capacity
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
    sha = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "command": command,
        "source_sha": sha,
        "gpu": args.gpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "omp_threads": 8,
        "runtime_environment": {
            key: value for key, value in environment.items() if key.startswith("DRAMA3_")
        },
        "benchmark_client_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/drama3/benchmark.py").read_bytes()
        ).hexdigest(),
        "source_status": subprocess.check_output(
            ["git", "-C", str(args.source), "status", "--porcelain"], text=True
        ),
    }
    (args.output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    with (args.output / "server.log").open("w") as server_log:
        process = subprocess.Popen(
            command,
            cwd=args.source,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def stop(_signal, _frame):
            raise KeyboardInterrupt("Benchmark interrupted; cleaning up its server")

        signal.signal(signal.SIGTERM, stop)
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = started + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {process.returncode}; see server.log")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server did not become ready within 30 minutes")
                time.sleep(2)
            metadata["startup_to_readiness_s"] = time.monotonic() - started
            (args.output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
            benchmark = [
                str(args.bench_python),
                "-m",
                "drama3.benchmark",
                "--url",
                url,
                "--backend",
                args.backend,
                "--corpus",
                str(args.corpus),
                "--output",
                str(args.output / "benchmark"),
                "--requests",
                str(args.requests),
                "--repeats",
                str(args.repeats),
                "--warm-references",
                "--save-audio",
                "--concurrency",
                *map(str, args.concurrency),
            ]
            (args.output / "benchmark-command.json").write_text(
                json.dumps(benchmark, indent=2) + "\n"
            )
            with (args.output / "benchmark.log").open("w") as benchmark_log:
                subprocess.run(
                    benchmark, check=True, stdout=benchmark_log, stderr=subprocess.STDOUT
                )
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
