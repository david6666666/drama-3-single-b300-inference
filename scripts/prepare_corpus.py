"""Extract a pinned Seed-TTS parquet into a portable JSONL request corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    args = parser.parse_args()
    parquet_path = hf_hub_download(
        "zhaochenyang20/seed-tts-eval-arrow",
        "data/en-00000-of-00001.parquet",
        repo_type="dataset",
        revision=args.revision,
    )
    table = pq.read_table(parquet_path)
    if not 0 < args.count <= table.num_rows:
        parser.error(f"Count must be between 1 and {table.num_rows}")
    args.output.mkdir(parents=True, exist_ok=False)
    references = args.output / "references"
    references.mkdir()
    rows = []
    for row in table.slice(0, args.count).to_pylist():
        audio = row["ref_audio"]["bytes"]
        digest = hashlib.sha256(audio).hexdigest()
        path = references / f"{digest}.wav"
        path.write_bytes(audio)
        rows.append(
            {
                "id": row["sample_id"],
                "text": row["target_text"],
                "reference_text": row["ref_text"],
                "reference_audio": str(path.resolve()),
                "reference_sha256": digest,
            }
        )
    (args.output / "requests.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "dataset": "zhaochenyang20/seed-tts-eval-arrow",
                "revision": args.revision,
                "parquet_sha256": hashlib.sha256(Path(parquet_path).read_bytes()).hexdigest(),
                "selection": f"first {args.count} English rows in parquet order",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
