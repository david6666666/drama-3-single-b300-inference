"""Isolate actual packed prefill and codec graph overlap from request scheduling."""

import argparse
import json
from pathlib import Path

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

from drama3.codec import load_codec, streaming_codec
from drama3.model import DramaModel
from drama3.packed_prefill import PackedPrefillGraph


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fish-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(s) for s in args.corpus.read_text().splitlines()][:8]
    model = DramaModel(str(args.checkpoint), slots=32, max_length=2048)
    assert not model.tuned_prefill_moe
    prompts = [
        build_drama3_prompt(
            s["text"],
            model.weights.tokenizer,
            model.config,
            [s["reference_text"]],
            [torch.tensor(s["reference_codes"])],
        )
        for s in samples[:6]
    ]
    capacity = triton.cdiv(sum(map(len, prompts)), 256) * 256
    packed = PackedPrefillGraph(model, 8, capacity)
    packed.run(prompts, list(range(6)))
    codec = streaming_codec(load_codec(args.checkpoint, args.fish_source), 64)
    stream = torch.cuda.Stream()
    tensors = [v for pair in (*model.states.values(), *model.kv.values()) for v in pair]
    tensors += [*codec.stream.tensors.values(), *codec.stream.kv_tensors.values()]
    initial = [v.clone() for v in tensors]
    results = []
    for batch, length, repeats in ((1, 6, 24), (4, 16, 8), (8, 64, 4)):
        codes = torch.stack(
            [torch.tensor(s["reference_codes"])[:, :length] for s in samples[:batch]]
        ).cuda()
        assert codes.shape == (batch, 10, length)
        rows = torch.tensor(
            [list(range(batch)), [1] * batch, [length] * batch, [0] * batch],
            device="cuda",
            dtype=torch.int64,
        )
        expected_state = expected_hidden = expected_pcm = None
        for concurrent in (False, True):
            for value, saved in zip(tensors, initial, strict=True):
                value.copy_(saved)
            begin, done = torch.cuda.Event(), torch.cuda.Event()
            begin.record()
            if concurrent:
                stream.wait_event(begin)
                with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(repeats):
                        pcm = codec(codes, rows=rows)
                    done.record()
                packed.graph.replay()
                torch.cuda.current_stream().wait_event(done)
            else:
                packed.graph.replay()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(repeats):
                        pcm = codec(codes, rows=rows)
            torch.cuda.synchronize()
            if not concurrent:
                expected_state = [v.clone() for v in tensors]
                expected_hidden = packed.output.clone()
                expected_pcm = pcm.clone()
            differences = [
                dict(
                    index=i,
                    shape=list(value.shape),
                    max_abs=float((value.float() - saved.float()).abs().max()),
                )
                for i, (value, saved) in enumerate(zip(tensors, expected_state, strict=True))
                if not torch.equal(value, saved)
            ]
            hidden_exact = torch.equal(packed.output, expected_hidden)
            pcm_exact = torch.equal(pcm, expected_pcm)
            result = dict(
                batch=batch,
                frames=length,
                codec_repeats=repeats,
                concurrent=concurrent,
                state_differences=differences,
                hidden_exact=hidden_exact,
                pcm_exact=pcm_exact,
                passed=not differences and hidden_exact and pcm_exact,
            )
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            assert result["passed"], "Concurrent prefill/codec changed native state or output"


if __name__ == "__main__":
    main()
