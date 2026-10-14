"""Compare the independent execution path against Fish Speech's native math."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext

from drama3.model import AttentionBatch, DramaModel


def compare(name, actual, expected, results, limit=0.03):
    a, b = actual.float().flatten(), expected.float().flatten()
    relative_rmse = ((a - b).square().mean() / b.square().mean().clamp_min(1e-12)).sqrt().item()
    row = {
        "name": name,
        "relative_rmse": relative_rmse,
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "max_abs": (a - b).abs().max().item(),
        "limit": limit,
        "pass": bool(torch.isfinite(a).all()) and relative_rmse <= limit,
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    if not row["pass"]:
        raise AssertionError(f"{name}: relative RMSE {relative_rmse} > {limit}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-moe", action="store_true", help="Check attention and tower only")
    parser.add_argument("--moe-batches", nargs="+", type=int, default=[1, 32])
    args = parser.parse_args()
    torch.manual_seed(123)
    model = DramaModel(args.checkpoint, slots=4, max_length=512)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for length in () if args.skip_moe else args.moe_batches:
            x = torch.randn(length, 2048, dtype=torch.bfloat16, device="cuda")
            for index in (0, 19, 39):
                layer = model.backbone.layers[index]
                compare(
                    f"moe/layer{index}/n{length}",
                    model.moe(x, layer.mlp, decode=True),
                    layer.mlp(x),
                    results,
                )
        for index in (0, 3, 19, 39):
            layer = model.backbone.layers[index]
            length = 32
            positions = torch.arange(length, device="cuda")
            slots = torch.zeros(length, dtype=torch.int64, device="cuda")
            x = torch.randn(length, 2048, dtype=torch.bfloat16, device="cuda")
            context = LayerContext(
                input_pos=positions,
                cos=model.cosine[positions].repeat(1, 2),
                sin=model.sine[positions].repeat(1, 2),
                kv_len=length,
                continue_state=False,
                use_cache=True,
            )
            if layer.layer_type == "full_attention":
                reference = layer.self_attn
                reference.allocate_cache(1, 512, torch.bfloat16, "cuda")
                expected = reference(x[None], context)[0]
                actual = model.attention(x, reference, index, positions, slots, None)
            else:
                reference = layer.linear_attn
                reference.allocate_cache(1, torch.bfloat16, "cuda")
                for state in model.states[index]:
                    state.zero_()
                expected = reference(x[None], context)[0]
                actual = model.linear_attention(x, reference, index, slots, False)
            compare(f"attention/layer{index}/prefill", actual, expected, results)
            for step in range(4):
                x = torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda")
                positions = torch.tensor([length + step], device="cuda")
                slots = torch.zeros(1, dtype=torch.int64, device="cuda")
                context = LayerContext(
                    input_pos=positions,
                    cos=model.cosine[positions].repeat(1, 2),
                    sin=model.sine[positions].repeat(1, 2),
                    kv_len=length + step + 1,
                    continue_state=True,
                    use_cache=True,
                )
                expected = reference(x[None], context)[0]
                if layer.layer_type == "full_attention":
                    batch = AttentionBatch(slots, positions, 512)
                    actual = model.attention(x, reference, index, positions, slots, batch)
                else:
                    actual = model.linear_attention(x, reference, index, slots, True)
                compare(f"attention/layer{index}/decode{step}", actual, expected, results)
        for index, layer in enumerate([*model.tower.shared_layers, *model.tower.residual_layers]):
            positions = torch.arange(32, device="cuda")
            slots = torch.zeros(32, dtype=torch.int64, device="cuda")
            x = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda")
            codes = torch.randint(4096, (32,), device="cuda")
            semantic = torch.ones(32, dtype=torch.bool, device="cuda")
            expected = layer(x, codes, semantic, positions)
            actual = model.tower_layer(
                x, codes, semantic, None, layer, index, positions, slots, None
            )
            compare(f"tower/layer{index}/prefill", actual, expected, results)
        print(
            "PASS: operator and layer checks; end-to-end quality remains a separate gate",
            flush=True,
        )
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
