"""Tune exact-quantization MoE row tiles using activations from real prompts."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.kernels import linear, native_moe
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--atomic", action="store_true")
    args = parser.parse_args()
    model = DramaModel(args.checkpoint, slots=4 * args.batch, max_length=512)
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][: args.batch]
    prompts = [
        build_drama3_prompt(
            s["text"],
            model.weights.tokenizer,
            model.config,
            [s["reference_text"]],
            [torch.tensor(s["reference_codes"])],
        )
        for s in samples
    ]
    span = max(128, triton.next_power_of_2(max(map(len, prompts))))
    frames = torch.zeros(args.batch, span, 11, dtype=torch.int64, device="cuda")
    frames[:, :, 0] = model.config.eos_token_id
    for row, prompt in enumerate(prompts):
        frames[row, : len(prompt)].copy_(prompt)
    original = model.moe
    indices = {id(layer.mlp): i for i, layer in enumerate(model.backbone.layers)}
    inputs = {}

    def capture(x, layer, decode=False):
        index = indices[id(layer)]
        if index in (0, 19, 39):
            inputs[index] = x.clone()
        return original(x, layer, decode)

    model.moe = capture
    model.prefill_batched(
        frames,
        torch.arange(args.batch, device="cuda"),
        torch.tensor(list(map(len, prompts)), device="cuda"),
    )
    model.moe = original
    results = []
    for index, x in inputs.items():
        layer = model.backbone.layers[index].mlp
        probability = linear(x, layer.gate.weight).float().softmax(-1)
        weights, ids = probability.topk(8, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        expected = native_moe(x, ids, weights, layer.experts)
        counts = torch.bincount(ids.flatten(), minlength=256).cpu()
        for block_m in (16, 32, 64, 128):
            for group_m in (1, 8):
                fn = partial(
                    native_moe, x, ids, weights, layer.experts, block_m, group_m, args.atomic
                )
                actual = fn()
                error = float(
                    (
                        (actual.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    ).sqrt()
                )
                record = dict(
                    batch=args.batch,
                    span=span,
                    layer=index,
                    prompt_lengths=list(map(len, prompts)),
                    max_expert_tokens=int(counts.max()),
                    block_m=block_m,
                    group_m=group_m,
                    atomic=args.atomic,
                    rrms=error,
                    passed=bool(actual.isfinite().all()) and error < 0.003,
                )
                if record["passed"]:
                    record["us"] = triton.testing.do_bench_cudagraph(fn) * 1000
                results.append(record)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
