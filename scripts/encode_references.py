"""Encode one shared corpus of reference codes for both measured runtimes.

FP32 reference encoding follows vLLM-Omni's production pre-encoding contract.
The resulting code integers are identical inputs to both servers.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from drama3.codec import load_codec


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    codec = load_codec(args.checkpoint, args.fish_source)
    cache = {}
    rows = []
    for row in map(json.loads, args.corpus.read_text().splitlines()):
        digest = hashlib.sha256(Path(row["reference_audio"]).read_bytes()).hexdigest()
        if digest not in cache:
            audio, rate = sf.read(row["reference_audio"], dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.mean(1)).cuda()[None, None]
            if rate != 44100:
                waveform = audio_functional.resample(waveform, rate, 44100)
            codes, _ = codec.encode(waveform, torch.tensor([waveform.shape[-1]], device="cuda"))
            cache[digest] = codes[0].cpu().tolist()
        row["reference_codes"] = cache[digest]
        rows.append(row)
    target = args.output / "requests.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metadata = {
        "input_corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "output_corpus_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "dtype": "float32",
        "encoder": "pinned Fish Speech codec",
        "checkpoint": str(args.checkpoint),
        "fish_source": str(args.fish_source),
        "requests": len(rows),
        "unique_references": len(cache),
        "preparation_s": time.perf_counter() - started,
    }
    (args.output / "encoding.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
