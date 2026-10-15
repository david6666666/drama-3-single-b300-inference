"""Render per-concurrency median metrics; never infer a missing comparison."""

import argparse
import json
import statistics
from pathlib import Path

METRICS = ["throughput_frames_s", "ttfa_p50_ms", "ttfa_p95_ms", "rtf_mean"]


def load_summary(path):
    rows = json.loads(path.read_text())
    groups = {}
    for row in rows:
        groups.setdefault(row["concurrency"], []).append(row)
    result = {}
    for concurrency, repeated in groups.items():
        if len(repeated) < 3 or any(not row["valid"] for row in repeated):
            raise ValueError(f"{path}, C{concurrency}: requires three valid repetitions")
        result[concurrency] = {
            name: statistics.median(row[name] for row in repeated) for name in METRICS
        }
        result[concurrency]["throughput_range"] = [
            min(row["throughput_frames_s"] for row in repeated),
            max(row["throughput_frames_s"] for row in repeated),
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", type=Path)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {
        name: load_summary(path)
        for name in ("sglang", "vllm", "candidate")
        if (path := getattr(args, name)) is not None
    }
    lines = [
        "# Measured streaming performance",
        "",
        "Median of three complete runs. TTFA is in milliseconds; throughput is acoustic frames/s.",
        "",
        "| Runtime | C | Frames/s | TTFA p50 | TTFA p95 | Mean RTF | Ratio to new vLLM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency in sorted({c for values in groups.values() for c in values}):
        baseline = groups.get("vllm", {}).get(concurrency, {}).get("throughput_frames_s")
        for name, values in groups.items():
            if concurrency not in values:
                continue
            row = values[concurrency]
            ratio = (
                f"{row['throughput_frames_s'] / baseline:.3f}×"
                if name == "candidate" and baseline
                else "—"
            )
            lines.append(
                f"| {name} | {concurrency} | {row['throughput_frames_s']:.2f} | "
                f"{row['ttfa_p50_ms']:.2f} | {row['ttfa_p95_ms']:.2f} | "
                f"{row['rtf_mean']:.4f} | {ratio} |"
            )
    lines += [
        "",
        "This table alone does not establish acceptance. Complete quality coverage, matched",
        "workloads and the recorded experiment controls must also pass. Missing evidence is",
        "INCONCLUSIVE, never PASS.",
        "",
    ]
    if "candidate" in groups:
        required = {1, 2, 4, 8, 16, 32, 64}
        covered = required <= groups["candidate"].keys() and required <= groups["vllm"].keys()
        if covered:
            passed = all(
                groups["candidate"][c]["throughput_frames_s"]
                >= 2 * groups["vllm"][c]["throughput_frames_s"]
                for c in required
            )
            verdict = "PASS" if passed else "FAIL"
        else:
            verdict = "INCOMPLETE"
        lines.extend(
            [f"Throughput-only 2× gate: **{verdict}**. Quality is a separate required gate.", ""]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
