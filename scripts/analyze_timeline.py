"""Analyze event spans within a raw benchmark interval, excluding its warmup."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def merged(rows):
    out = []
    for left, right in sorted(rows):
        if out and left <= out[-1][1]:
            out[-1][1] = max(out[-1][1], right)
        else:
            out.append([left, right])
    return out


def overlap(a, b):
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    start = min(r["dispatch_s"] for r in requests)
    end = max(r["dispatch_s"] + r["completion_s"] for r in requests)
    source = json.loads(args.timeline.read_text())
    assert source["nonblocking_events_diagnostics_only"]
    rows = [r for r in source["records"] if start <= r["cpu_start_s"] <= end]
    groups = defaultdict(list)
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ("gpu_start_ms", "gpu_end_ms", "gpu_ms"))
        assert row["gpu_start_ms"] >= 0 and row["gpu_end_ms"] >= row["gpu_start_ms"]
        assert row["gpu_ms"] >= 0 and row["cpu_submit_ms"] >= 0
        groups[row["stage"]].append(row)
    spans = {
        stage: merged((r["gpu_start_ms"], r["gpu_end_ms"]) for r in values)
        for stage, values in groups.items()
    }
    stages = {
        stage: dict(
            calls=len(values),
            mean_active_batch=statistics.mean(r["batch"] for r in values),
            summed_event_ms=sum(r["gpu_ms"] for r in values),
            cpu_submit_ms=sum(r["cpu_submit_ms"] for r in values),
        )
        for stage, values in groups.items()
    }
    codec = spans.get("codec_gpu_and_copy", [])
    decode = spans.get("decode", [])
    prefill = spans.get("prefill", [])
    by_batch = defaultdict(list)
    for row in groups["decode"]:
        shared = overlap([[row["gpu_start_ms"], row["gpu_end_ms"]]], codec)
        by_batch[(row["batch"], bool(shared))].append(row["gpu_ms"])
    result = dict(
        diagnostics_only=True,
        requests=len(requests),
        raw_request_interval_s=end - start,
        stage_records=len(rows),
        excluded_records=len(source["records"]) - len(rows),
        stages=stages,
        codec_decode_overlap_ms=overlap(codec, decode),
        codec_prefill_overlap_ms=overlap(codec, prefill),
        codec_event_union_ms=sum(right - left for left, right in codec),
        decode_batches=[
            dict(
                active=batch,
                overlaps_codec=shared,
                calls=len(values),
                mean_event_ms=statistics.mean(values),
            )
            for (batch, shared), values in sorted(by_batch.items())
        ],
        limitations=[
            "CUDA event spans can include GPU waiting for host submission; "
            "these are not SM utilization or summed kernel durations.",
            "Overlap and slower decode are correlated observations; batches, context lengths "
            "and tail steps differ, so this is not causal attribution.",
            "Active batch counts do not count graph multi-steps and cannot be converted "
            "directly to generated frames.",
            "Warmup is excluded by the shared host monotonic clock, "
            "not by choosing favorable measured requests.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "decode_batches"}, indent=2))


if __name__ == "__main__":
    main()
