"""Check padding at individual layers, then graph replay against the same eager shape.

Changing GEMM row count can change BF16 rounding and downstream MoE routing.
End-to-end quality, rather than hidden-state identity across row counts, is a
separate required gate.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from drama3.engine import PrefillGraph
from drama3.model import DramaModel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = DramaModel(args.checkpoint, slots=4, max_length=512)
    graph = None if args.diagnose else PrefillGraph(model, 128)
    results = []
    for length in (33, 91, 128):
        frames = torch.zeros((length, 11), dtype=torch.int64, device="cuda")
        frames[:, 0] = torch.randint(1000, 10000, (length,), device="cuda")
        frames[5:20, 1:] = torch.randint(4096, (15, 10), device="cuda")
        frames[5:20, 0] = model.config.semantic_start_token_id + frames[5:20, 1]
        if args.diagnose:
            padded = torch.nn.functional.pad(frames, (0, 0, 0, 128 - length))
            hidden = model.tower.embed_backbone(frames, model.backbone.embed_tokens(frames[:, 0]))
            other = model.tower.embed_backbone(padded, model.backbone.embed_tokens(padded[:, 0]))
            for conv, state in model.states.values():
                conv.zero_()
                state.zero_()
            for index, layer in enumerate(model.backbone.layers):
                mixed = []
                for slot, current in enumerate((hidden, other)):
                    normalized = model.norm(current, layer.input_layernorm)
                    positions = torch.arange(len(current), device="cuda")
                    slots = torch.full_like(positions, slot)
                    value = (
                        model.attention(normalized, layer.self_attn, index, positions, slots, None)
                        if layer.layer_type == "full_attention"
                        else model.linear_attention(
                            normalized,
                            layer.linear_attn,
                            index,
                            slots,
                            False,
                            torch.tensor([length], device="cuda") if slot else None,
                        )
                    )
                    current = current + value
                    current = current + model.moe(
                        model.norm(current, layer.post_attention_layernorm), layer.mlp
                    )
                    mixed.append(current)
                hidden, other = mixed
                rrms = (
                    (
                        (hidden.float() - other[:length].float()).square().mean()
                        / hidden.float().square().mean()
                    )
                    .sqrt()
                    .item()
                )
                print("LAYER", length, index, layer.layer_type, rrms, flush=True)
            continue
        padded = torch.nn.functional.pad(frames, (0, 0, 0, 128 - length))
        positions = torch.arange(128, device="cuda")
        slots = torch.zeros_like(positions)
        valid_length = torch.tensor([length], device="cuda")
        for index in (0, 1, 19, 39):
            layer = model.backbone.layers[index]
            x = torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16)
            if layer.layer_type == "linear_attention":
                conv, state = model.states[index]
                conv.zero_()
                state.zero_()
                reference = model.linear_attention(
                    x[:length], layer.linear_attn, index, slots[:length], False
                )
                actual = model.linear_attention(
                    x, layer.linear_attn, index, slots + 1, False, valid_length
                )[:length]
                checks = [(actual, reference), (conv[1], conv[0]), (state[1], state[0])]
            else:
                reference = model.attention(
                    x[:length], layer.self_attn, index, positions[:length], slots[:length], None
                )
                actual = model.attention(x, layer.self_attn, index, positions, slots + 1, None)[
                    :length
                ]
                checks = [(actual, reference)]
            for check_index, (a, b) in enumerate(checks):
                rrms = (
                    (
                        (a.float() - b.float()).square().mean()
                        / b.float().square().mean().clamp_min(1e-12)
                    )
                    .sqrt()
                    .item()
                )
                print("PADDING", length, index, check_index, rrms, flush=True)
                assert rrms < 0.03, ("padding layer", length, index, check_index, rrms)
        # Padding data is arbitrary and must not affect the causal valid prefix.
        graph.frames.zero_()
        expected = model.prefill_padded(padded, positions, slots, valid_length).clone()
        actual = graph.run(frames, 1)
        torch.cuda.synchronize()
        errors = []
        pairs = [("hidden", actual, expected)]
        for index, (conv, state) in model.states.items():
            pairs.extend(
                [(f"conv{index}", conv[1], conv[0]), (f"state{index}", state[1], state[0])]
            )
        for index, (keys, values) in model.kv.items():
            stop = length if index < 40 else length - 1
            pairs.extend(
                [
                    (f"k{index}", keys[1, :stop], keys[0, :stop]),
                    (f"v{index}", values[1, :stop], values[0, :stop]),
                ]
            )
        for name, a, b in pairs:
            rrms = (
                (
                    (a.float() - b.float()).square().mean()
                    / b.float().square().mean().clamp_min(1e-12)
                )
                .sqrt()
                .item()
            )
            errors.append({"name": name, "relative_rmse": rrms})
            assert torch.isfinite(a).all() and rrms < 0.03, (length, name, rrms)
        started = time.perf_counter()
        for _ in range(10):
            graph.run(frames, 1)
        torch.cuda.synchronize()
        row = {
            "length": length,
            "graph_ms": (time.perf_counter() - started) * 100,
            "errors": errors,
            "pass": True,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
