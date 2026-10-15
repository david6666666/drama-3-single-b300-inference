"""Check exact native quantization buffers, including strided inputs and zeros."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from fish_speech.models.text2semantic.fish_omni.fp8_moe import quant_kernel, quantize_groups

from drama3.moe_input import prepare_moe_input


def original_prepare(x, router):
    q, scales = quantize_groups(quant_kernel, x, 2048)
    return q, scales.T.contiguous(), router.float()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for batch in (12, 16, 24, 32, 40, 48, 56, 64):
        for strided in (False, True):
            for magnitude in (0.0, 1e-12, 1.0, 1000.0):
                storage = torch.randn(
                    batch, 4096 if strided else 2048, device="cuda", dtype=torch.bfloat16
                ).mul_(magnitude)
                x = storage[:, :2048]
                router = torch.randn(
                    batch, 1408 if strided else 256, device="cuda", dtype=torch.bfloat16
                )[:, :256]

                baseline = partial(original_prepare, x, router)
                expected = baseline()
                actual = prepare_moe_input(x, router)
                exact = [
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(actual, expected, strict=True)
                ]
                row = dict(
                    batch=batch,
                    strided=strided,
                    magnitude=magnitude,
                    buffers_exact=exact,
                    passed=all(exact),
                )
                if row["passed"] and magnitude == 1.0 and strided:
                    row["baseline_us"] = triton.testing.do_bench_cudagraph(baseline) * 1000
                    row["fused_us"] = (
                        triton.testing.do_bench_cudagraph(partial(prepare_moe_input, x, router))
                        * 1000
                    )
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                assert row["passed"], "Native FP8 quantization or router conversion changed"


if __name__ == "__main__":
    main()
