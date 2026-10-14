"""Copy compact benchmark evidence without redistributing evaluation waveforms."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(args.source.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".yaml", ".log"):
            continue
        if "hf_modules" in path.parts:
            continue
        relative = path.relative_to(args.source)
        data = path.read_bytes()
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".jsonl", ".log") or len(data) > 1024 * 1024:
            target = target.with_name(target.name + ".gz")
            target.write_bytes(gzip.compress(data, mtime=0))
        else:
            target.write_bytes(data)
        records.append(
            {
                "source": str(relative),
                "artifact": str(target.relative_to(args.destination)),
                "uncompressed_sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    (args.destination / "files.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
