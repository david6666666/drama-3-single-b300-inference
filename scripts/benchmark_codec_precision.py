"""Compare native and streaming codec precision on identical reference codes."""

import argparse
import json
from pathlib import Path

import torch

from drama3.codec import bake_weight_norm, load_codec
from drama3.ops.dac_channels_last import ChannelsLastDACDecoder, DACDecodeGraphs, DACStreamState


def rrms(actual, expected):
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
    args = parser.parse_args()
    sample = json.loads(args.corpus.read_text().splitlines()[0])
    codes = torch.tensor(sample["reference_codes"], device="cuda")[:, :64].unsqueeze(0)
    results = []
    for name, dtype in (
        ("current_fp32_weights_bf16_compute", torch.bfloat16),
        ("fp16", torch.float16),
    ):
        codec = load_codec(args.checkpoint, args.fish_source)
        reference = codec.from_indices(codes).float()
        bake_weight_norm(codec)
        if name == "fp16":
            codec.to(dtype)
        with torch.autocast("cuda", dtype=dtype):
            native = codec.from_indices(codes).float()
        decoder = ChannelsLastDACDecoder(codec, dtype=dtype)
        decoder.compile_transformer()
        graphs = DACDecodeGraphs(
            decoder,
            10,
            (6, 8, 12, 16, 24, 32, 64),
            512,
            16,
            receptive_frames=0,
            stream=DACStreamState(16),
        )
        with torch.autocast("cuda", dtype=dtype):
            graphs.capture(torch.device("cuda"))
            chunks, position = [], 0
            windows = iter((6, 8, 12, 16, 24, 32))
            while position < codes.shape[-1]:
                length = min(next(windows, 64), codes.shape[-1] - position)
                rows = torch.tensor(
                    [[0], [int(position == 0)], [length], [position]], device="cuda"
                )
                chunks.append(graphs(codes[:, :, position : position + length], rows=rows).clone())
                position += length
        streamed = torch.cat(chunks, dim=-1).float()
        assert bool(streamed.isfinite().all()), dtype
        row = dict(
            configuration=name,
            dtype=str(dtype),
            frames=codes.shape[-1],
            native_vs_fp32_rrms=rrms(native, reference),
            stream_vs_fp32_rrms=rrms(streamed, reference),
            stream_vs_native_rrms=rrms(streamed, native),
            replay_ms={str(key): value for key, value in graphs.cost_ms.items()},
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del codec, decoder, graphs, chunks, streamed, native, reference


if __name__ == "__main__":
    main()
