"""Aggregate complete, validated matrix records; never launches or initializes CUDA."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
BACKENDS = ("standalone", "vllm", "sglang")
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
PERF = ("throughput_frames_s", "ttfa_p50_ms", "ttfa_p95_ms", "rtf_mean")


def require(okay, message):
    if not okay:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), "invalid three-run values")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def aggregate(protocol_path, manifest_path):
    protocol, manifest = read(protocol_path), read(manifest_path)
    protocol_hash, manifest_hash = digest(protocol_path), digest(manifest_path)
    require(protocol["concurrency"] == list(CONCURRENCY) and protocol["repeats"] == 3
            and protocol["requests"] == 1088, "wrong matrix contract")
    require(set(protocol["frameworks"]) == set(BACKENDS), "wrong framework coverage")
    matrix_plan, matrix_manifest = read(OUT / "matrix-protocol.json"), read(OUT / "matrix-frozen-inputs.json")
    matrix_protocol_hash, matrix_manifest_hash = digest(OUT / "matrix-protocol.json"), digest(OUT / "matrix-frozen-inputs.json")
    require(matrix_plan["base_protocol_sha256"] == protocol_hash
            and matrix_plan["base_manifest_sha256"] == manifest_hash, "matrix base differs")
    require(manifest["files"] and matrix_manifest, "empty frozen manifest")
    for entries in (manifest["files"], matrix_manifest):
        for path, record in entries.items():
            require(Path(path).stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                    "frozen input changed: " + path)
    labels = {f"c{c}-r{r}" for c in CONCURRENCY for r in range(3)}
    client_hash = digest(OUT / "source-standalone/src/drama3/benchmark.py")
    rows, provenance, audits = [], {}, {}
    for backend in BACKENDS:
        root = OUT / "matrix" / backend
        result, launch = read(root / "result.json"), read(root / "service/launch.json")
        require(result["status"] == "COMPLETE_VALID_SERVICE_AND_QUALITY", backend + " incomplete")
        require(result["protocol_sha256"] == protocol_hash and result["manifest_sha256"] == manifest_hash,
                backend + " belongs to another protocol")
        require(result["matrix_protocol_sha256"] == matrix_protocol_hash
                and result["matrix_manifest_sha256"] == matrix_manifest_hash, backend + " matrix protocol differs")
        require(launch["source_sha"] == protocol["source_shas"][backend]
                and not launch["source_status"] and launch["benchmark_client_sha256"] == client_hash,
                backend + " source/client mismatch")
        performance, quality = result["service"], result["quality"]
        for audit in (performance, quality):
            require(audit["status"] == "PASS" and audit["groups_checked"] == 21
                    and audit["requests_checked"] == 22848, backend + " incomplete audit")
            require(len(audit["groups"]) == 21 and {g["label"] for g in audit["groups"]} == labels,
                    backend + " missing or duplicate run labels")
            require(all(g["requests"] == 1088 for g in audit["groups"]), backend + " short run")
        require(performance["all_wav_pcm_hashes_verified"] and quality["unfiltered_wer"]
                and quality["all_sim_records_present"], backend + " incomplete audio/quality coverage")
        require(performance["corpus_sha256"] == protocol["corpus_sha256"]
                and performance["summary_sha256"] == digest(root / "service/benchmark/summary.json")
                and quality["quality_sha256"] == digest(root / "quality.json"), backend + " audit drift")
        for group in performance["groups"]:
            require(group["jsonl_sha256"] == digest(root / "service/benchmark" / (group["label"] + ".jsonl")),
                    backend + " raw records changed: " + group["label"])
        pg = {g["label"]: g for g in performance["groups"]}
        qg = {g["label"]: g for g in quality["groups"]}
        audits[backend] = qg
        provenance[backend] = {"source_sha": launch["source_sha"], "launch": launch,
                               "result_sha256": digest(root / "result.json"),
                               "performance_audit": performance, "quality_audit": quality}
        for c in CONCURRENCY:
            selected = [f"c{c}-r{r}" for r in range(3)]
            metrics = {key: stats([pg[label][key] for label in selected]) for key in PERF}
            metrics.update({key: stats([qg[label][key] for label in selected])
                            for key in ("wer_percent", "sim_mean")})
            rows.append({"framework": backend, "concurrency": c, "repeats": 3,
                         "requests_per_repeat": 1088, "metrics": metrics})
    for row in rows:
        c, backend = row["concurrency"], row["framework"]
        baseline = next(x for x in rows if x["framework"] == "vllm" and x["concurrency"] == c)
        dw = row["metrics"]["wer_percent"]["median"] - baseline["metrics"]["wer_percent"]["median"]
        ds = row["metrics"]["sim_mean"]["median"] - baseline["metrics"]["sim_mean"]["median"]
        differences = [{"repeat": r,
                        "wer_delta_pp": audits[backend][f"c{c}-r{r}"]["wer_percent"] - audits["vllm"][f"c{c}-r{r}"]["wer_percent"],
                        "sim_delta": audits[backend][f"c{c}-r{r}"]["sim_mean"] - audits["vllm"][f"c{c}-r{r}"]["sim_mean"]} for r in range(3)]
        row["quality_vs_vllm"] = {"wer_median_delta_pp": dw, "sim_median_delta": ds,
                                  "median_within_reporting_bounds": dw <= 1 and ds >= -2,
                                  "paired_repeat_differences": differences,
                                  "all_repeats_within_reporting_bounds": all(x["wer_delta_pp"] <= 1 and x["sim_delta"] >= -2 for x in differences)}
    preparation = read(OUT / "preparation.json")
    return {"status": "COMPLETE_DESCRIPTIVE_COMPARISON", "optimization_acceptance_evaluated": False,
            "aggregation": "Independent medians of three run metrics; min/max are observed ranges, not confidence intervals. Repeat labels do not imply identical RNG/output lengths.",
            "quality_reporting_bounds": {"wer_increase_pp": 1, "sim_decrease": 2, "filters_results": False},
            "protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash,
            "matrix_protocol_sha256": matrix_protocol_hash, "matrix_manifest_sha256": matrix_manifest_hash,
            "client_sha256": client_hash, "generator_sha256": digest(__file__),
            "recorded_dependency_versions": {b: preparation["imports"][b]["packages"] for b in BACKENDS},
            "preparation_sha256": digest(OUT / "preparation.json"), "provenance": provenance,
            "performance_groups": 63, "quality_groups": 63, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.protocol, args.manifest)
    with (OUT / "matrix-summary.json").open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
