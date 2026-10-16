"""CPU-only audit of the exact five saved SGLang groups, including one failure.

Validation success describes preserved evidence, never a complete matrix PASS.
Records are stored in completion order; request indices bind the corpus order.
"""

import hashlib
import importlib.util
import json
import statistics
import wave
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "_comparison_full_validation", ART / "validate_results.py"
)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
require, finite, close = full.require, full.finite, full.close
digest, read_records, read_corpus = full.digest, full.read_records, full.read_corpus
EXPECTED_GROUPS = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
LABELS = [f"c{c}-r{r}" for c, r in EXPECTED_GROUPS]
FAILED_LABEL = "c2-r1"
FAILED_ID = "c2-r1-00351"
FAILED_SAMPLE = "common_voice_en_19770519-common_voice_en_19770516"
FAILED_ERROR = 'HTTP 500: {"error":{"message":"Request speech-f8b5c838-1439-46ed-af6d-ab20f6d13079: Drama 3 generated no audio frames","type":"server_error","param":null,"code":null}}'
CORPUS_SHA = "08612ee05ab0dd8a762109978b5f230e5b0311a1dfde87f03a4572434dff2dac"


def _identity(rows, samples, label, count=1088):
    require(len(rows) == count, label + ": record count")
    expected = {
        f"{label}-{i:05d}": samples[i % len(samples)]["id"] for i in range(count)
    }
    require(
        len({r["request_id"] for r in rows}) == count, label + ": duplicate request"
    )
    require(
        {r["request_id"]: r["sample_id"] for r in rows} == expected,
        label + ": request/corpus identity",
    )


def _saved_groups(output_dir, corpus):
    bench = Path(output_dir) / "benchmark"
    samples = read_corpus(corpus)
    require(digest(corpus) == CORPUS_SHA, "not the frozen corpus")
    manifest = json.loads((bench / "manifest.json").read_text())
    require(
        manifest["backend"] == "sglang",
        "only the documented SGLang interruption is accepted",
    )
    require(manifest["corpus_sha256"] == CORPUS_SHA, "manifest corpus differs")
    require(
        manifest["requests"] == 1088
        and manifest["repeats"] == 3
        and manifest["concurrency"] == [1, 2, 4, 8, 16, 32, 64],
        "original 21-group plan changed",
    )
    require(
        manifest["max_frames"] == 500
        and manifest["warmup"] == 8
        and manifest["save_audio"] is True
        and manifest["warm_references"] is False,
        "benchmark controls changed",
    )
    require(
        manifest["connection_policy"]
        == "fresh TCP connection per request; no automatic retries",
        "connection policy changed",
    )
    summaries = json.loads((bench / "summary.json").read_text())
    require(
        [(s["concurrency"], s["repeat"]) for s in summaries] == EXPECTED_GROUPS,
        "expected exactly five ordered groups, without omissions or retries",
    )
    require(
        {p.name for p in bench.glob("*.jsonl")}
        == {label + ".jsonl" for label in LABELS},
        "extra/missing request group",
    )
    require(
        {p.name for p in bench.glob("*-warmup.json")}
        == {"c1-warmup.json", "c2-warmup.json"},
        "unexpected warmup groups",
    )
    groups = []
    for summary, label in zip(summaries, LABELS, strict=True):
        rows = read_records(bench / (label + ".jsonl"))
        _identity(rows, samples, label)
        failures = [r for r in rows if r["success"] is not True]
        require(
            len(failures) == int(label == FAILED_LABEL),
            label + ": unexpected failure count",
        )
        if failures:
            row = failures[0]
            require(
                row["success"] is False
                and row["request_id"] == FAILED_ID
                and row["sample_id"] == FAILED_SAMPLE
                and row["http_status"] == 500
                and row["error"] == FAILED_ERROR,
                "original HTTP 500 failure changed",
            )
            require(
                not (bench / (FAILED_ID + ".wav")).exists(),
                "failed request must not acquire invented audio",
            )
            require(
                not any(
                    k in row for k in ("audio_samples", "audio_sha256", "ttfa_s", "rtf")
                ),
                "failed request has invented success fields",
            )
        groups.append((label, summary, rows))
    return bench, samples, groups


def _audio_and_timing(row, bench):
    rid = row["request_id"]
    require(
        row["success"] is True and row["http_status"] == 200 and not row.get("error"),
        rid + ": unexpected service failure",
    )
    for key in (
        "dispatch_s",
        "ttfa_s",
        "last_audio_s",
        "completion_s",
        "rtf",
        "rms",
        "peak",
    ):
        finite(row[key], rid + " " + key, 0)
    require(
        row["ttfa_s"] <= row["last_audio_s"] <= row["completion_s"],
        rid + ": time ordering",
    )
    arrivals, sizes = row["chunk_arrivals_s"], row["chunk_bytes"]
    require(
        len(arrivals) == len(sizes) == row["transport_chunks"] > 0, rid + ": chunks"
    )
    require(
        arrivals == sorted(arrivals)
        and all(isinstance(x, int) and x > 0 for x in sizes),
        rid + ": chunk sequence",
    )
    for value in arrivals:
        finite(value, rid + " chunk arrival", 0)
    close(row["ttfa_s"], arrivals[0], rid + " TTFA")
    close(row["last_audio_s"], arrivals[-1], rid + " last audio")
    with wave.open(str(bench / (rid + ".wav")), "rb") as wav:
        require(
            (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getcomptype(),
            )
            == (44100, 1, 2, "NONE"),
            rid + ": WAV format",
        )
        require(
            wav.getnframes() == row["audio_samples"] > 0, rid + ": WAV sample count"
        )
        pcm = wav.readframes(wav.getnframes())
    require(sum(sizes) == len(pcm) == row["audio_samples"] * 2, rid + ": PCM length")
    require(hashlib.sha256(pcm).hexdigest() == row["audio_sha256"], rid + ": PCM hash")
    close(row["acoustic_frames"], row["audio_samples"] / 2048, rid + " acoustic frames")
    close(row["audio_duration_s"], row["audio_samples"] / 44100, rid + " duration")
    close(row["rtf"], row["last_audio_s"] / row["audio_duration_s"], rid + " RTF")


def validate_saved_service(output_dir, corpus):
    """Verify 5439 successful WAVs plus the unchanged one HTTP500 record."""
    bench, samples, groups = _saved_groups(output_dir, corpus)
    for concurrency in (1, 2):
        label = f"c{concurrency}-warmup"
        rows = json.loads((bench / (label + ".json")).read_text())
        _identity(rows, samples, label, 8)
        require(
            all(r["success"] is True and r["http_status"] == 200 for r in rows),
            label + ": warmup failed",
        )
    expected_wavs = {
        r["request_id"] + ".wav" for _, _, rows in groups for r in rows if r["success"]
    }
    require(
        {p.name for p in bench.glob("c*-r*.wav")} == expected_wavs,
        "extra/missing measured WAV",
    )
    checked = []
    for label, summary, rows in groups:
        successes = [r for r in rows if r["success"]]
        failed = 1088 - len(successes)
        require(
            summary["requests"] == 1088
            and summary["completed"] == len(successes)
            and summary["failed"] == failed
            and summary["valid"] is (failed == 0),
            label + ": summary completeness",
        )
        elapsed = finite(summary["makespan_s"], label + " makespan", 1e-12)
        for row in rows:
            finite(row["dispatch_s"], row["request_id"] + " dispatch", 0)
            finite(row["completion_s"], row["request_id"] + " completion", 0)
            if row["success"]:
                _audio_and_timing(row, bench)
        # Do not require completion-ordered JSONL to be numerically sorted by ID.
        dispatched = sorted(rows, key=lambda r: r["request_id"])
        require(
            [r["dispatch_s"] for r in dispatched]
            == sorted(r["dispatch_s"] for r in dispatched),
            label + ": request IDs no longer bind dispatch/corpus order",
        )
        observed = max(r["dispatch_s"] + r["completion_s"] for r in rows) - min(
            r["dispatch_s"] for r in rows
        )
        require(
            elapsed + 1e-7 >= observed, label + ": makespan excludes failure or tail"
        )
        frames = sum(r["audio_samples"] for r in successes) / 2048
        metrics = {
            "throughput_frames_s": frames / elapsed,
            "ttfa_p50_ms": full.percentile([r["ttfa_s"] for r in successes], 0.5)
            * 1000,
            "ttfa_p95_ms": full.percentile([r["ttfa_s"] for r in successes], 0.95)
            * 1000,
            "rtf_mean": statistics.mean(r["rtf"] for r in successes),
        }
        for name, value in metrics.items():
            finite(summary[name], label + " " + name, 0)
            close(summary[name], value, label + " " + name)
        checked.append(
            {
                "label": label,
                "requests": 1088,
                "successes": len(successes),
                "failed": failed,
                "valid": failed == 0,
                "diagnostic_only": failed != 0,
                "frames": frames,
                "makespan_s": elapsed,
                "observed_request_span_s": observed,
                "jsonl_sha256": digest(bench / (label + ".jsonl")),
                **metrics,
            }
        )
    return {
        "status": "VALIDATED_WITH_KNOWN_PARTIAL_FAILURE",
        "full_matrix_valid": False,
        "planned_groups": 21,
        "groups_checked": 5,
        "complete_groups": 4,
        "partial_groups": 1,
        "requests_checked": 5440,
        "successful_audio_checked": 5439,
        "failed_records": 1,
        "all_successful_wav_pcm_hashes_verified": True,
        "known_failure": {
            "request_id": FAILED_ID,
            "sample_id": FAILED_SAMPLE,
            "http_status": 500,
            "error": FAILED_ERROR,
        },
        "corpus_sha256": digest(corpus),
        "manifest_sha256": digest(bench / "manifest.json"),
        "summary_sha256": digest(bench / "summary.json"),
        "groups": checked,
    }


def _check_quality_group(group, rows, samples, label):
    records = group["records"]
    require(len(records) == 1088, label + ": quality record count")
    require(
        [(r["request_id"], r["sample_id"]) for r in records]
        == [(r["request_id"], r["sample_id"]) for r in rows],
        label + ": quality identity/order differs from saved service",
    )
    scored = []
    for service, record in zip(rows, records, strict=True):
        rid = record["request_id"]
        require(
            record["target_text"] == samples[record["sample_id"]]["text"],
            rid + ": target text",
        )
        if not service["success"]:
            require(
                record["is_success"] is False
                and record["error"] == service["error"]
                and record["speaker_similarity"] is None,
                rid + ": original failure/null SIM not preserved",
            )
            require(
                all(
                    record[k] == 0
                    for k in ("wer", "substitutions", "deletions", "insertions", "hits")
                ),
                rid + ": failed request has invented WER scores",
            )
            require(
                all(record[k] == "" for k in ("whisper_text", "ref_norm", "hyp_norm")),
                rid + ": failed request has invented transcription",
            )
            continue
        require(
            record["is_success"] is True and not record["error"],
            rid + ": successful audio was not scored",
        )
        finite(record["wer"], rid + " WER", 0)
        finite(record["speaker_similarity"], rid + " SIM")
        for key in ("substitutions", "deletions", "insertions", "hits"):
            require(
                type(record[key]) is int and record[key] >= 0, rid + ": edit counts"
            )
        errors = record["substitutions"] + record["deletions"] + record["insertions"]
        words = record["hits"] + record["substitutions"] + record["deletions"]
        close(
            record["wer"],
            errors / words if words else float(record["insertions"]),
            rid + " per-record WER",
        )
        scored.append(record)
    count = len(scored)
    skipped = 1088 - count
    require(count == 1088 - int(label == FAILED_LABEL), label + ": wrong scored count")
    wer, sim = group["wer"], group["similarity"]
    require(
        group["complete"] is (skipped == 0)
        and wer["total_samples"] == sim["total"] == 1088
        and wer["evaluated"] == sim["evaluated"] == count
        and wer["skipped"] == sim["skipped"] == skipped,
        label + ": partial coverage misrepresented",
    )
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in scored)
    words = sum(r["hits"] + r["substitutions"] + r["deletions"] for r in scored)
    require(words > 0, label + ": no reference words")
    values = [r["wer"] for r in scored]
    expected = {
        "wer_corpus": errors / words,
        "wer_per_sample_mean": statistics.mean(values),
        "wer_per_sample_median": statistics.median(values),
        "wer_per_sample_std": statistics.pstdev(values),
        "wer_per_sample_p95": full.percentile(values, 0.95),
        "wer_per_sample_max": max(values),
    }
    for key, value in expected.items():
        finite(wer[key], label + " " + key, 0)
        close(wer[key], value, label + " " + key)
    finite(sim["mean"], label + " mean SIM")
    close(
        sim["mean"],
        statistics.mean(r["speaker_similarity"] for r in scored),
        label + " unfiltered SIM",
    )
    return {
        "label": label,
        "requests": 1088,
        "scored_requests": count,
        "skipped": skipped,
        "complete": skipped == 0,
        "diagnostic_only": skipped != 0,
        "wer_percent": wer["wer_corpus"] * 100,
        "sim_mean": sim["mean"],
        "errors": errors,
        "reference_words": words,
        "failure_preserved": skipped != 0,
    }


def validate_saved_quality(path, service_dir, corpus):
    """Audit four complete scores and one deliberately incomplete diagnostic group."""
    _, samples, groups = _saved_groups(service_dir, corpus)
    samples = {r["id"]: r for r in samples}
    data = json.loads(Path(path).read_text())
    require(
        list(data["groups"]) == LABELS,
        "expected all five quality groups in evaluator order",
    )
    checked = [
        _check_quality_group(data["groups"][label], rows, samples, label)
        for label, _, rows in groups
    ]
    return {
        "status": "VALIDATED_WITH_KNOWN_PARTIAL_FAILURE",
        "full_matrix_valid": False,
        "groups_checked": 5,
        "complete_groups": 4,
        "partial_groups": 1,
        "requests_checked": 5440,
        "scored_requests": 5439,
        "failed_records": 1,
        "unfiltered_wer": True,
        "all_successful_sim_records_present": True,
        "known_failure_sim": None,
        "quality_sha256": digest(path),
        "corpus_sha256": digest(corpus),
        "service_summary_sha256": digest(Path(service_dir) / "benchmark/summary.json"),
        "groups": checked,
    }
