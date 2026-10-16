"""Audit actual saved service and adversarial CPU-only partial-score fixtures."""

import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

import validators as v

OUT = Path(__file__).resolve().parent
ART = OUT.parent
SERVICE = ART / "interrupted-matrix-report-01/sglang/service"
CORPUS = Path(
    "/home/zjy/code/hsliu/artifacts/drama3-b300-20261014/seedtts-en1088-codes/requests.jsonl"
)
checks = []


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")


def check(name, value):
    checks.append({"name": name, "passed": bool(value)})
    v.require(value, name)


def rejects(name, fn):
    try:
        fn()
    except (RuntimeError, KeyError, ValueError):
        check(name, True)
    else:
        check(name, False)


def replace(path, data, jsonl=False):
    if path.is_symlink():
        path.unlink()
    path.write_text(
        ("\n".join(json.dumps(x) for x in data) + "\n") if jsonl else json.dumps(data)
    )


def synthetic_quality(groups, samples):
    result = {"asr": {"synthetic_CPU_fixture": True}, "groups": {}}
    for label, _, rows in groups:
        records = []
        for index, row in enumerate(rows):
            record = {
                "request_id": row["request_id"],
                "sample_id": row["sample_id"],
                "target_text": samples[row["sample_id"]]["text"],
                "is_success": row["success"],
                "error": "",
                "whisper_text": "",
                "ref_norm": "",
                "hyp_norm": "",
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
                "speaker_similarity": None,
            }
            if row["success"]:
                words = 1 + index % 7
                substitutions = words if index % 100 == 0 else 0
                insertions = 2 if index % 100 == 0 else 0
                record.update(
                    whisper_text="synthetic",
                    ref_norm="synthetic",
                    hyp_norm="synthetic",
                    substitutions=substitutions,
                    insertions=insertions,
                    hits=words - substitutions,
                    wer=(substitutions + insertions) / words,
                    speaker_similarity=20.0 + index % 3,
                )
            else:
                record["error"] = row["error"]
            records.append(record)
        scored = [r for r in records if r["is_success"]]
        values = [r["wer"] for r in scored]
        errors = sum(
            r["substitutions"] + r["deletions"] + r["insertions"] for r in scored
        )
        words = sum(r["substitutions"] + r["deletions"] + r["hits"] for r in scored)
        result["groups"][label] = {
            "records": records,
            "complete": len(scored) == 1088,
            "wer": {
                "lang": "en",
                "total_samples": 1088,
                "evaluated": len(scored),
                "skipped": 1088 - len(scored),
                "wer_corpus": errors / words,
                "wer_per_sample_mean": statistics.mean(values),
                "wer_per_sample_median": statistics.median(values),
                "wer_per_sample_std": statistics.pstdev(values),
                "wer_per_sample_p95": v.full.percentile(values, 0.95),
                "wer_per_sample_max": max(values),
            },
            "similarity": {
                "total": 1088,
                "evaluated": len(scored),
                "skipped": 1088 - len(scored),
                "mean": statistics.mean(r["speaker_similarity"] for r in scored),
            },
        }
    return result


def main():
    audit = v.validate_saved_service(SERVICE, CORPUS)
    save(OUT / "saved-service-audit.json", audit)
    check(
        "actual_saved_service_4full_1partial_5439WAV",
        audit["complete_groups"] == 4
        and audit["partial_groups"] == 1
        and audit["successful_audio_checked"] == 5439
        and audit["failed_records"] == 1,
    )
    bench, samples, groups = v._saved_groups(SERVICE, CORPUS)
    with tempfile.TemporaryDirectory(prefix="saved-sglang-validator-") as tmp:
        fixture = Path(tmp)
        b = fixture / "benchmark"
        b.mkdir()
        for path in bench.iterdir():
            (b / path.name).symlink_to(path)
        summaries = json.loads((bench / "summary.json").read_text())
        for name, changed in [
            ("group_omission", summaries[:-1]),
            ("group_reordered", summaries[::-1]),
            ("extra_group", summaries + [dict(summaries[-1], repeat=2)]),
        ]:
            replace(b / "summary.json", changed)
            rejects(name, lambda: v._saved_groups(fixture, CORPUS))
        replace(b / "summary.json", summaries)
        partial = copy.deepcopy(groups[-1][2])
        cases = {}
        cases["missing_record"] = partial[:-1]
        cases["duplicate_id"] = copy.deepcopy(partial)
        cases["duplicate_id"][0]["request_id"] = cases["duplicate_id"][1]["request_id"]
        cases["sample_id_changed"] = copy.deepcopy(partial)
        cases["sample_id_changed"][0]["sample_id"] = "wrong"
        cases["extra_failure"] = copy.deepcopy(partial)
        cases["extra_failure"][0]["success"] = False
        cases["missing_original_failure"] = copy.deepcopy(partial)
        failure_index = next(i for i, r in enumerate(partial) if not r["success"])
        cases["missing_original_failure"][failure_index]["success"] = True
        cases["failure_error_changed"] = copy.deepcopy(partial)
        cases["failure_error_changed"][failure_index]["error"] = "different HTTP500"
        for name, rows in cases.items():
            replace(b / "c2-r1.jsonl", rows, jsonl=True)
            rejects(name, lambda: v._saved_groups(fixture, CORPUS))
        replace(b / "c2-r1.jsonl", partial, jsonl=True)
        (b / "c4-r0.jsonl").write_text("")
        rejects("unexpected_later_group", lambda: v._saved_groups(fixture, CORPUS))
        (b / "c4-r0.jsonl").unlink()
        changed = copy.deepcopy(summaries)
        changed[0]["throughput_frames_s"] *= 1.01
        replace(b / "summary.json", changed)
        rejects(
            "service_metric_mismatch", lambda: v.validate_saved_service(fixture, CORPUS)
        )
        replace(b / "summary.json", summaries)
        changed = copy.deepcopy(summaries)
        changed[0]["makespan_s"] = 1
        replace(b / "summary.json", changed)
        rejects(
            "service_tail_exclusion", lambda: v.validate_saved_service(fixture, CORPUS)
        )
        replace(b / "summary.json", summaries)
        changed = copy.deepcopy(summaries)
        changed[-1].update(valid=True, completed=1088, failed=0)
        replace(b / "summary.json", changed)
        rejects(
            "partial_summary_marked_full",
            lambda: v.validate_saved_service(fixture, CORPUS),
        )
        replace(b / "summary.json", summaries)
        quality = synthetic_quality(groups, {r["id"]: r for r in samples})
        quality_path = fixture / "quality-synthetic.json"
        save(quality_path, quality)
        qa = v.validate_saved_quality(quality_path, fixture, CORPUS)
        check(
            "synthetic_4full_1partial_quality_preserved",
            qa["scored_requests"] == 5439
            and qa["partial_groups"] == 1
            and qa["groups"][-1]["complete"] is False,
        )
        check(
            "poor_quality_scores_are_reported_not_rejected",
            qa["groups"][0]["sim_mean"] < 22,
        )
        mutations = []
        bad = copy.deepcopy(quality)
        del bad["groups"][v.FAILED_LABEL]
        mutations.append(("quality_group_omission", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["records"].pop()
        mutations.append(("quality_missing_failure_record", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["complete"] = True
        mutations.append(("quality_partial_marked_full", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["wer"]["evaluated"] = 1088
        mutations.append(("quality_invented_evaluated_count", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["similarity"]["skipped"] = 0
        mutations.append(("quality_hidden_skipped_failure", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["records"][failure_index]["is_success"] = True
        mutations.append(("quality_failed_record_marked_success", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["records"][failure_index][
            "speaker_similarity"
        ] = 0.0
        mutations.append(("quality_invented_failure_SIM", bad))
        bad = copy.deepcopy(quality)
        bad["groups"][v.FAILED_LABEL]["records"][failure_index]["error"] = ""
        mutations.append(("quality_lost_original_error", bad))
        bad = copy.deepcopy(quality)
        bad["groups"]["c1-r0"]["records"][0]["speaker_similarity"] = float("nan")
        mutations.append(("quality_nan_SIM", bad))
        bad = copy.deepcopy(quality)
        bad["groups"]["c1-r0"]["records"][0]["request_id"] = "bad"
        mutations.append(("quality_wrong_ID", bad))
        bad = copy.deepcopy(quality)
        bad["groups"]["c2-r0"]["records"][:2] = bad["groups"]["c2-r0"]["records"][1::-1]
        mutations.append(("quality_reordered_records", bad))
        bad = copy.deepcopy(quality)
        bad["groups"]["c1-r0"]["wer"]["wer_corpus"] += 0.01
        mutations.append(("quality_WER_aggregate_mismatch", bad))
        bad = copy.deepcopy(quality)
        bad["groups"]["c1-r0"]["similarity"]["mean"] += 1
        mutations.append(("quality_SIM_aggregate_mismatch", bad))
        for name, bad in mutations:
            save(quality_path, bad)
            rejects(
                name, lambda: v.validate_saved_quality(quality_path, fixture, CORPUS)
            )
    check(
        "no_torch_cuda_import", "torch" not in sys.modules and "cuda" not in sys.modules
    )
    result = {
        "status": "CPU_SAVED_PARTIAL_VALIDATORS_PASS",
        "checks": checks,
        "gpu_used": False,
        "actual_service_audit": "saved-service-audit.json",
        "quality_scope": "Synthetic adversarial CPU fixtures only; actual saved outputs are scored separately by the original evaluator.",
        "validator_sha256": v.digest(OUT / "validators.py"),
        "test_sha256": v.digest(Path(__file__)),
        "full_helper_sha256": v.digest(ART / "validate_results.py"),
        "saved_service_audit_sha256": v.digest(OUT / "saved-service-audit.json"),
    }
    save(OUT / "validator-cpu-check.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(checks),
                "validator_sha256": result["validator_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
