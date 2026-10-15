"""Validate output and true-end convolution state before prefill timings."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from drama3.prefill_conv import prefill_conv


def reference(x, weight, state, slots, span, lengths):
    sequences = len(x) // span
    sequence = x.reshape(sequences, span, 8192).transpose(1, 2).contiguous()
    indices = lengths[:, None] - 3 + torch.arange(3, device=x.device)
    state[slots] = sequence.gather(-1, indices[:, None, :].expand(-1, 8192, -1))
    mixed = F.silu(F.conv1d(sequence, weight, padding=3, groups=8192))
    return mixed[..., :span].transpose(1, 2).reshape(-1, 8192).contiguous()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    torch.manual_seed(42)
    for batch in (1, 4, 8):
        for span in (128, 256, 512):
            x = torch.randn(batch * span, 12352, device="cuda", dtype=torch.bfloat16)[:, :8192]
            weight = torch.randn(8192, 1, 4, device="cuda", dtype=x.dtype)
            slots = torch.arange(2 * batch - 1, batch - 1, -1, device="cuda")
            lengths = torch.linspace(3, span, batch, device="cuda").long()
            expected_state = torch.randn(2 * batch, 8192, 3, device="cuda", dtype=x.dtype)
            actual_state = expected_state.clone()
            old = partial(reference, x, weight, expected_state, slots, span, lengths)
            new = partial(prefill_conv, x, weight, actual_state, slots, span, lengths)
            expected, actual = old(), new()
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
            error = float(
                (
                    (actual.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            assert bool(actual.isfinite().all()) and error < 0.003, error
            result = dict(
                batch=batch,
                span=span,
                rrms=error,
                reference_us=triton.testing.do_bench_cudagraph(old) * 1000,
                triton_us=triton.testing.do_bench_cudagraph(new) * 1000,
            )
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
