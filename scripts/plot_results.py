"""Export the four requested metrics only after a matched, complete comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402
from report_results import METRICS, load_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", type=Path, required=True, help="Baseline run directory")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate run directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"vLLM-Omni": args.vllm, "Standalone": args.candidate}
    manifests = [json.loads((p / "benchmark/manifest.json").read_text()) for p in paths.values()]
    for field in (
        "corpus_sha256",
        "concurrency",
        "requests",
        "repeats",
        "warmup",
        "max_frames",
        "warm_references",
        "frame_measure",
        "rtf_boundary",
        "connection_policy",
    ):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f"Unmatched workload control: {field}")
    launches = [json.loads((p / "launch.json").read_text()) for p in paths.values()]
    for field in ("gpu", "cpu_affinity", "omp_threads"):
        if launches[0][field] != launches[1][field]:
            raise ValueError(f"Unmatched hardware control: {field}")
    concurrency = [1, 2, 4, 8, 16, 32, 64]
    summaries = {
        name: load_summary(path / "benchmark/summary.json") for name, path in paths.items()
    }
    if any(set(summary) != set(concurrency) for summary in summaries.values()):
        raise ValueError("All seven concurrency points are required")
    raw = {
        name: json.loads((path / "benchmark/summary.json").read_text())
        for name, path in paths.items()
    }
    for name, rows in raw.items():
        for c in concurrency:
            repeats = [r["repeat"] for r in rows if r["concurrency"] == c]
            if sorted(repeats) != [0, 1, 2]:
                raise ValueError(f"{name}, C{c}: exactly repetitions 0, 1 and 2 are required")
    titles = ["Throughput (frames/s)", "TTFA p50 (ms)", "TTFA p95 (ms)", "Mean RTF"]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7), layout="constrained")
    for axis, metric, title in zip(axes.flat, METRICS, titles, strict=True):
        for (name, summary), color in zip(summaries.items(), ("#d77525", "#2475b0"), strict=True):
            values = [summary[c][metric] for c in concurrency]
            low = [min(r[metric] for r in raw[name] if r["concurrency"] == c) for c in concurrency]
            high = [max(r[metric] for r in raw[name] if r["concurrency"] == c) for c in concurrency]
            axis.plot(
                concurrency, values, "o-", color=color, label=name, linewidth=1.8, markersize=4
            )
            axis.fill_between(concurrency, low, high, color=color, alpha=0.15)
        if metric == "throughput_frames_s":
            axis.plot(
                concurrency,
                [2 * summaries["vLLM-Omni"][c][metric] for c in concurrency],
                "--",
                color="#666666",
                label="2× baseline target",
            )
            axis.set_yscale("log", base=2)
            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            axis.legend(frameon=False, fontsize=9)
        else:
            axis.set_ylim(bottom=0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(concurrency, concurrency)
        axis.set_xlabel("Concurrency")
        axis.set_title(title, loc="left", fontsize=12)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Drama 3 · one B300 · {manifests[0]['requests']} requests/run\n"
        "Three-run medians; shading shows the repetition range",
        fontsize=13,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    figure.savefig(args.output / "comparison.png", dpi=180)
    figure.savefig(args.output / "comparison.pdf", metadata={"CreationDate": None})
    plt.close(figure)
    (args.output / "data.json").write_text(
        json.dumps(
            {
                "source_sha": {
                    name: launch["source_sha"] for name, launch in zip(paths, launches, strict=True)
                },
                "metrics": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
