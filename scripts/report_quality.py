"""Report complete matched WER/SIM coverage without inferring missing scores."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(paths):
    groups, settings = {}, None
    for path in paths:
        document = json.loads(path.read_text())
        if settings is not None and settings != document["asr"]:
            raise ValueError(f"ASR settings differ: {path}")
        settings = document["asr"]
        for label, group in document["groups"].items():
            if label in groups:
                raise ValueError(f"Duplicate quality group: {label}")
            records, wer, similarity = group["records"], group["wer"], group.get("similarity", {})
            count = len(records)
            if not (
                group.get("complete")
                and count > 0
                and wer["evaluated"] == wer["total_samples"] == count
                and similarity.get("evaluated") == similarity.get("total") == count
                and wer["skipped"] == similarity.get("skipped") == 0
            ):
                raise ValueError(f"Incomplete quality coverage: {path}, {label}")
            samples = sorted((r["sample_id"], r["target_text"]) for r in records)
            groups[label] = dict(
                wer_percent=wer["wer_corpus"] * 100,
                similarity=similarity["mean"],
                count=count,
                fingerprint=hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
            )
    return groups, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, base_settings = load(args.vllm)
    candidate, candidate_settings = load(args.candidate)
    if base_settings != candidate_settings:
        raise ValueError("ASR settings must match across runtimes")
    required = {f"c{c}-r0" for c in (1, 2, 4, 8, 16, 32, 64)} | {"c64-r1", "c64-r2"}
    for label in required:
        if label not in baseline or label not in candidate:
            raise ValueError(f"Missing required quality group: {label}")
        if baseline[label]["fingerprint"] != candidate[label]["fingerprint"]:
            raise ValueError(f"Evaluation samples differ: {label}")
    lines = [
        "# Full-corpus audio quality",
        "",
        "First performance repetition at each concurrency. WER is corpus WER in percent;",
        "SIM is the upstream WavLM speaker-similarity score. Every row includes the full",
        "corpus; high-WER outliers are retained. These are measurements, not user-approved",
        "quality thresholds.",
        "",
        "| C | Samples/runtime | vLLM WER % | Standalone WER % | Δ WER pp | "
        "vLLM SIM | Standalone SIM | Δ SIM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in (1, 2, 4, 8, 16, 32, 64):
        base, current = baseline[f"c{c}-r0"], candidate[f"c{c}-r0"]
        lines.append(
            f"| {c} | {base['count']} | {base['wer_percent']:.3f} | {current['wer_percent']:.3f} | "
            f"{current['wer_percent'] - base['wer_percent']:+.3f} | {base['similarity']:.3f} | "
            f"{current['similarity']:.3f} | {current['similarity'] - base['similarity']:+.3f} |"
        )
    lines.extend(["", "C64 was scored for all three repetitions:", ""])
    for name, values in (("vLLM-Omni", baseline), ("Standalone", candidate)):
        errors = [values[f"c64-r{r}"]["wer_percent"] for r in range(3)]
        similarities = [values[f"c64-r{r}"]["similarity"] for r in range(3)]
        lines.append(
            f"- {name}: median WER {statistics.median(errors):.3f}% "
            f"(range {min(errors):.3f}–{max(errors):.3f}%), median SIM "
            f"{statistics.median(similarities):.3f} "
            f"(range {min(similarities):.3f}–{max(similarities):.3f})."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
