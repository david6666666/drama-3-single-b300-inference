"""B300 operator correctness checks and isolated latency measurements."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.tower import sample_top_k_top_p

from drama3.kernels import conv_update, sample


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for batch in (1, 8, 64):
        state = torch.randn(128, 8192, 3, device="cuda", dtype=torch.bfloat16)
        saved = state.clone()
        slots = torch.randperm(128, device="cuda")[:batch]
        x = torch.randn(batch, 8256, device="cuda", dtype=torch.bfloat16)[:, :8192]
        weight = torch.randn(8192, 1, 4, device="cuda", dtype=torch.bfloat16)
        window = torch.cat((saved[slots], x.unsqueeze(-1)), dim=-1)
        expected = F.silu(F.conv1d(window, weight, groups=8192)).squeeze(-1)
        actual = conv_update(x, state, weight, slots)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        torch.testing.assert_close(state[slots], window[..., 1:], atol=0, rtol=0)
        untouched = torch.ones(128, dtype=torch.bool, device="cuda")
        untouched[slots] = False
        torch.testing.assert_close(state[untouched], saved[untouched], atol=0, rtol=0)
        results.append(
            {
                "operator": "conv_update",
                "batch": batch,
                "pass": True,
                "max_abs_error": (actual - expected).abs().max().item(),
                "latency_ms": triton.testing.do_bench(
                    partial(conv_update, x, state, weight, slots)
                ),
            }
        )
        for width in (4096, 4097):
            logits = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
            step = torch.zeros((), dtype=torch.int64, device="cuda")
            probabilities = logits.float().softmax(-1)
            sorted_probabilities, ids = probabilities.sort(descending=True)
            permitted = sorted_probabilities.cumsum(-1) <= 0.9
            permitted[:, 30:] = False
            permitted[:, 0] = True
            mask = torch.zeros_like(permitted).scatter(1, ids, permitted)
            draws = []
            for _ in range(256):
                step.add_(1)
                draw = sample(logits, step, 0)
                assert mask.gather(1, draw[:, None]).all(), (
                    "Sample left the allowed top-p/top-k support"
                )
                draws.append(draw)
            assert torch.unique(torch.stack(draws)).numel() > 1, "RNG did not advance"
            fused_ms = triton.testing.do_bench(partial(sample, logits, step, 0))
            reference_ms = triton.testing.do_bench(
                partial(sample_top_k_top_p, logits, 1.0, 30, 0.9, check_finite=False)
            )
            results.append(
                {
                    "operator": "sample",
                    "batch": batch,
                    "width": width,
                    "pass": True,
                    "fused_ms": fused_ms,
                    "reference_ms": reference_ms,
                }
            )
        print(json.dumps(results[-3:]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
