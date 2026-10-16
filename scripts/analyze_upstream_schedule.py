"""Summarize CPU scheduler-step annotations in a saved upstream Torch trace."""

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
    }


def union_ms(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1000


def summarize(rows):
    return {
        "steps": len(rows),
        "kind_counts": dict(Counter(row["kind"] for row in rows)),
        "cpu_annotation_span_sum_ms": sum(row["cpu_span_ms"] for row in rows),
        "prompt_requests_scheduled": sum(row["prompt_requests"] for row in rows),
        "prompt_tokens_scheduled": sum(row["prompt_tokens"] for row in rows),
        "decode_requests_scheduled": sum(row["decode_requests"] for row in rows),
        "decode_tokens_scheduled": sum(row["decode_tokens"] for row in rows),
        "scheduled_requests": distribution([row["scheduled_requests"] for row in rows]),
        "mixed_prompt_requests": distribution(
            [row["prompt_requests"] for row in rows if row["kind"] == "mixed"]
        ),
        "mixed_prompt_tokens": distribution(
            [row["prompt_tokens"] for row in rows if row["kind"] == "mixed"]
        ),
        "mixed_decode_requests": distribution(
            [row["decode_requests"] for row in rows if row["kind"] == "mixed"]
        ),
        "pure_decode_requests": distribution(
            [row["decode_requests"] for row in rows if row["kind"] == "decode"]
        ),
    }


def analyze(trace, metadata=None, gpu_spans=False):
    events = trace["traceEvents"]
    rows = []
    gpu = defaultdict(list)
    for event in events:
        match = STEP.fullmatch(event.get("name", ""))
        if not match or event.get("ph") != "X":
            continue
        category = event.get("cat")
        external_id = event.get("args", {}).get("External id")
        if category == "gpu_user_annotation" and gpu_spans and external_id is not None:
            gpu[(external_id, event["name"])].append((event["ts"], event["ts"] + event["dur"]))
        if category != "user_annotation":
            continue
        prompt, prompt_tokens, decode, decode_tokens = map(int, match.groups())
        kind = (
            "mixed"
            if prompt and decode
            else "prefill"
            if prompt
            else "decode"
            if decode
            else "empty"
        )
        rows.append(
            {
                "kind": kind,
                "prompt_requests": prompt,
                "prompt_tokens": prompt_tokens,
                "decode_requests": decode,
                "decode_tokens": decode_tokens,
                "scheduled_requests": prompt + decode,
                "cpu_start_us": event["ts"],
                "cpu_span_ms": event["dur"] / 1000,
                "external_id": external_id,
                "name": event["name"],
            }
        )
    rows.sort(key=lambda row: row["cpu_start_us"])
    if not rows:
        raise ValueError("No CPU user_annotation scheduler steps found in this trace")
    if any(not math.isfinite(row["cpu_span_ms"]) or row["cpu_span_ms"] < 0 for row in rows):
        raise ValueError("Invalid CPU annotation duration")
    phases = defaultdict(list)
    requests = (metadata or {}).get("requests", [])
    anchor = (metadata or {}).get("profile_started_monotonic_s")
    can_align = anchor is not None and "baseTimeNanoseconds" in trace and bool(requests)
    for row in rows:
        if gpu_spans:
            spans = gpu.get((row["external_id"], row["name"]), [])
            row["gpu_annotation_fragments"] = len(spans)
            row["gpu_annotation_union_ms"] = union_ms(spans)
            row["gpu_annotation_envelope_ms"] = (
                (max(end for _, end in spans) - min(start for start, _ in spans)) / 1000
                if spans
                else None
            )
        if can_align:
            unix = trace["baseTimeNanoseconds"] / 1e9 + row["cpu_start_us"] / 1e6
            host = anchor + unix - metadata["profile_started_unix"]
            dispatched = sum(request["dispatch_s"] <= host for request in requests)
            completed = sum(
                request["dispatch_s"] + request["completion_s"] <= host for request in requests
            )
            row["client_requests_in_flight"] = dispatched - completed
            phase = (
                "initial_fill"
                if dispatched < metadata["concurrency"]
                else "continuous_admission"
                if dispatched < len(requests)
                else "final_drain"
            )
            row["client_phase"] = phase
            phases[phase].append(row)
    result = {
        "diagnostic_only": True,
        "counting_rule": "One CPU user_annotation execute_context event = one scheduler step",
        "summary": summarize(rows),
        "by_kind": {
            kind: summarize([row for row in rows if row["kind"] == kind])
            for kind in sorted({row["kind"] for row in rows})
        },
        "client_phase_alignment_available": can_align,
        "by_client_phase": {phase: summarize(values) for phase, values in phases.items()},
        "request_metadata": {
            key: metadata[key]
            for key in ("concurrency", "profile_request_count", "corpus_sha256")
            if metadata and key in metadata
        },
        "observed_client_requests": len(requests),
        "single_wave_workload": (len(requests) <= metadata["concurrency"] if requests else None),
        "steps": rows,
        "limitations": [
            "PIECEWISE graph launches and GPU annotation fragments are not scheduler steps.",
            "Prompt request counts include scheduled chunks, not necessarily distinct requests; "
            "sample IDs and request IDs remain in the separate request metadata.",
            "CPU annotation spans measure host execution scopes, not GPU latency or utilization.",
            "Optional GPU spans join by External id and event name and union overlapping "
            "annotations; envelopes include gaps and are not summed kernel time or SM utilization.",
            "Client phases use adjacent wall/monotonic clock samples and profiler timestamps. "
            "They distinguish ongoing admissions from final drain, not a verified steady state; "
            "client in-flight counts include queueing and codec work.",
            "Whole-run averages include initial fill and final drain. A single-wave trace "
            "cannot characterize sustained continuous batching.",
            "Profiler overhead changes timings; this analysis is not acceptance evidence.",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--requests", type=Path, help="profile_upstream cN.json request metadata")
    parser.add_argument("--gpu-spans", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as stream:
        trace = json.load(stream)
    metadata = json.loads(args.requests.read_text()) if args.requests else None
    result = analyze(trace, metadata, args.gpu_spans)
    result["trace_path"] = str(args.trace.resolve())
    result["trace_sha256"] = sha256(args.trace)
    if args.requests:
        result["request_metadata_sha256"] = sha256(args.requests)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "steps"}, indent=2))


if __name__ == "__main__":
    main()
