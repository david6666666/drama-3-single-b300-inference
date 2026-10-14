"""Score saved benchmark audio using upstream Seed-TTS WER and speaker SIM.

The Whisper checkpoint is the upstream-supported large-v3 model, run offline
with greedy decoding. Metric implementation and speaker network come directly
from the frozen SGLang-Omni source supplied with --upstream.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def read_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if rate != 16000:
        divisor = np.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    return audio


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["c1-r0", "c64-r0"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.upstream.resolve()))
    from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
    from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
    from benchmarks.tasks.asr import apply_wer

    assets = {row["repo"]: row["path"] for row in json.loads(args.assets.read_text())}
    corpus = {row["id"]: row for row in map(json.loads, args.corpus.read_text().splitlines())}
    groups = {}
    for label in args.labels:
        rows = [
            json.loads(line)
            for line in (args.benchmark / f"{label}.jsonl").read_text().splitlines()
        ]
        for row in rows:
            row["wav"] = str(args.benchmark / f"{row['request_id']}.wav")
        groups[label] = rows
    whisper = assets["openai/whisper-large-v3"]
    processor = AutoProcessor.from_pretrained(whisper)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper, dtype=torch.float16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    result = {
        "asr": {"checkpoint": whisper, "language": "en", "num_beams": 1, "max_new_tokens": 256},
        "groups": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for label, rows in groups.items():
        outputs = []
        result["groups"][label] = {"records": []}
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            valid = [row for row in batch if row["success"] and Path(row["wav"]).is_file()]
            hypotheses = {}
            if valid:
                waveforms = [read_audio(row["wav"]) for row in valid]
                if any(len(audio) > 30 * 16000 for audio in waveforms):
                    raise ValueError("Whisper 30-second window exceeded; chunking required")
                inputs = processor(
                    waveforms, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
                )
                inputs = {
                    name: tensor.to(
                        "cuda", dtype=torch.float16 if tensor.is_floating_point() else tensor.dtype
                    )
                    for name, tensor in inputs.items()
                }
                tokens = model.generate(
                    **inputs,
                    language="english",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=256,
                )
                texts = processor.batch_decode(tokens, skip_special_tokens=True)
                hypotheses = {
                    row["request_id"]: text for row, text in zip(valid, texts, strict=True)
                }
            for row in batch:
                output = SampleOutput(
                    sample_id=row["sample_id"],
                    target_text=corpus[row["sample_id"]]["text"],
                    is_success=row["request_id"] in hypotheses,
                )
                if output.is_success:
                    apply_wer(output, hypotheses[row["request_id"]], "en")
                else:
                    output.error = row.get("error") or "Missing generated WAV"
                outputs.append(output)
                result["groups"][label]["records"].append(
                    {
                        **dataclasses.asdict(output),
                        "request_id": row["request_id"],
                        "speaker_similarity": None,
                    }
                )
            save()
            print(f"WER {label}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
        result["groups"][label]["wer"] = calculate_wer_metrics(outputs, "en")
        save()
    del model, processor
    torch.cuda.empty_cache()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets["popsoda2002/seedtts-wavlm-sim"],
        wavlm_base=assets["s3prl/converted_ckpts"],
        device="cuda:0",
    )
    for label, rows in groups.items():
        records = result["groups"][label]["records"]
        scoreable = [
            (row, record) for row, record in zip(rows, records, strict=True) if record["is_success"]
        ]
        for start in range(0, len(scoreable), args.batch_size):
            batch = scoreable[start : start + args.batch_size]
            scores = scorer.score_batch(
                [corpus[row["sample_id"]]["reference_audio"] for row, _ in batch],
                [row["wav"] for row, _ in batch],
            )
            for (_, record), score in zip(batch, scores, strict=True):
                record["speaker_similarity"] = score
            save()
            print(
                f"SIM {label}: {min(start + len(batch), len(scoreable))}/{len(scoreable)}",
                flush=True,
            )
        scores = [
            record["speaker_similarity"]
            for record in records
            if record["speaker_similarity"] is not None
        ]
        result["groups"][label]["similarity"] = {
            "mean": float(np.mean(scores)) if scores else None,
            "evaluated": len(scores),
            "total": len(records),
            "skipped": len(records) - len(scores),
        }
        result["groups"][label]["complete"] = len(scores) == len(rows)
        save()


if __name__ == "__main__":
    main()
