"""Compare channels-last and stateful codec outputs with the original Fish codec."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from drama3.codec import bake_weight_norm, load_codec
from drama3.ops.dac_channels_last import ChannelsLastDACDecoder, DACDecodeGraphs, DACStreamState


def relative_rmse(actual, expected):
    return float(
        (
            (actual.float() - expected.float()).square().mean()
            / expected.float().square().mean().clamp_min(1e-12)
        ).sqrt()
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    codec = load_codec(args.checkpoint, args.fish_source)
    sample = json.loads(args.corpus.read_text().splitlines()[0])
    audio, rate = sf.read(sample["reference_audio"], dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(1)).cuda()[None, None]
    if rate != 44100:
        audio = audio_functional.resample(audio, rate, 44100)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        codes = codec.encode(audio, torch.tensor([audio.shape[-1]], device="cuda"))[0]
        expected = codec.from_indices(codes).float()
    bake_weight_norm(codec)
    decoder = ChannelsLastDACDecoder(codec)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        whole = decoder(codes).float()
    results = [{"check": "whole_vs_fish", "relative_rmse": relative_rmse(whole, expected)}]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results[-1]), flush=True)
    stream = DACStreamState(3)
    graphs = None
    if args.graphs:
        decoder.compile_transformer()
        graphs = DACDecodeGraphs(
            decoder,
            10,
            (6, 8, 12, 16, 24, 32, 64),
            128,
            2,
            receptive_frames=0,
            stream=stream,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            graphs.capture(torch.device("cuda"))
    for repetition in range(2):
        # Second pass reuses dirty slots and exercises a fresh request reset.
        chunks = []
        position = 0
        windows = iter([6, 8, 12, 16, 24, 32])
        while position < codes.shape[-1]:
            length = min(next(windows, 64), codes.shape[-1] - position)
            chunk = codes[:, :, position : position + length].expand(2, -1, -1)
            rows = torch.tensor(
                [
                    [2, 0],
                    [int(position == 0)] * 2,
                    [length] * 2,
                    [position] * 2,
                ],
                device="cuda",
                dtype=torch.int64,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if graphs is None:
                    stream.bind(rows, length)
                    decoder.stream = stream
                    output = decoder(chunk).clone()
                    decoder.stream = None
                    stream.unbind()
                else:
                    output = graphs(chunk, rows=rows)
            chunks.append(output)
            position += length
        actual = torch.cat(chunks, dim=-1)
        for name, reference in (("fish", expected), ("whole", whole)):
            row = {
                "check": f"stream_vs_{name}",
                "reuse": repetition,
                "graphs": args.graphs,
                "relative_rmse": relative_rmse(actual, reference.expand_as(actual)),
            }
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    assert all(row["relative_rmse"] < 0.03 for row in results), results


if __name__ == "__main__":
    main()
