"""Validate and tune FP32 GDN state updates, without changing model precision."""

# These closures run to completion within each batch iteration.
# ruff: noqa: B023

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for batch in (1, 8, 32, 64):
        mixed = torch.randn(batch, 8192, device="cuda", dtype=torch.bfloat16)
        a, b = torch.randn(2, batch, 32, device="cuda", dtype=torch.bfloat16)
        log = torch.randn(32, device="cuda")
        bias = torch.randn(32, device="cuda")
        slots = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int64)
        initial = torch.randn(batch, 32, 128, 128, device="cuda")
        state = initial.clone()
        output = torch.empty(batch, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
        reference = partial(
            fused_recurrent_gated_delta_rule_packed_decode,
            mixed,
            a,
            b,
            log,
            bias,
            128**-0.5,
            state,
            output,
            slots,
            True,
        )
        reference()
        expected, expected_state = output.clone(), state.clone()

        def tuned(tile, warps):
            fused_recurrent_gated_delta_rule_packed_decode_kernel[(128 // tile, batch * 32)](
                mixed_qkv=mixed,
                a=a,
                b=b,
                A_log=log,
                dt_bias=bias,
                o=output,
                h0=state,
                ht=state,
                ssm_state_indices=slots,
                scale=128**-0.5,
                stride_mixed_qkv_tok=mixed.stride(0),
                stride_a_tok=a.stride(0),
                stride_b_tok=b.stride(0),
                stride_init_state_token=state.stride(0),
                stride_final_state_token=state.stride(0),
                stride_indices_seq=slots.stride(0),
                H=16,
                HV=32,
                K=128,
                V=128,
                BK=128,
                BV=tile,
                SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=True,
                num_warps=warps,
                num_stages=3,
            )

        def flashinfer():
            q, k, v = mixed.split((2048, 2048, 4096), dim=-1)
            return gated_delta_rule_decode_pretranspose(
                q.reshape(batch, 1, 16, 128).contiguous(),
                k.reshape(batch, 1, 16, 128).contiguous(),
                v.reshape(batch, 1, 32, 128).contiguous(),
                None,
                log,
                a[:, None],
                bias,
                b[:, None],
                output=output,
                initial_state=state,
                initial_state_indices=slots,
                output_state_indices=slots,
                backend="flashinfer",
            )

        choices = [("sglang", reference), ("flashinfer", flashinfer)]
        choices += [
            (f"tile{tile}-warps{warps}", partial(tuned, tile, warps))
            for tile in (8, 16, 32, 64)
            for warps in (1, 4, 8)
        ]
        for name, fn in choices:
            state.copy_(initial)
            fn()
            error = float(
                (
                    (output.float() - expected.float()).square().mean()
                    / expected.float().square().mean()
                ).sqrt()
            )
            state_error = float(
                ((state - expected_state).square().mean() / expected_state.square().mean()).sqrt()
            )
            elapsed = triton.testing.do_bench_cudagraph(fn)
            row = {
                "batch": batch,
                "backend": name,
                "output_rrms": error,
                "state_rrms": state_error,
                "microseconds": elapsed * 1000,
                "pass": error < 0.003 and state_error < 0.003,
            }
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
