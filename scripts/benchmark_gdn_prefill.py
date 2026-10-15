"""Compare FLA and FlashInfer prefill outputs and indexed recurrent states."""

import argparse
import json
from pathlib import Path

import torch
import triton
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

from drama3.gdn import prefill


def rrms(a, b):
    return float(
        (
            (a.float() - b.float()).square().mean() / b.float().square().mean().clamp_min(1e-12)
        ).sqrt()
    )


def recurrent_fp32(q, k, v, g, beta, lengths):
    """Independent recurrence after the shared BF16 Q/K normalization boundary."""
    q = l2norm_fwd(q.contiguous()).float().repeat_interleave(2, dim=2)
    k = l2norm_fwd(k.contiguous()).float().repeat_interleave(2, dim=2)
    state = torch.zeros(len(q), 32, 128, 128, device=q.device)
    outputs = []
    for step in range(q.shape[1]):
        decayed = state * g[:, step, :, None, None].exp()
        residual = v[:, step].float() - (decayed * k[:, step, :, None, :]).sum(-1)
        updated = (
            decayed + (residual * beta[:, step, :, None])[:, :, :, None] * k[:, step, :, None, :]
        )
        state = torch.where((step < lengths)[:, None, None, None], updated, state)
        outputs.append((state * q[:, step, :, None, :]).sum(-1) * (128**-0.5))
    return torch.stack(outputs, dim=1), state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-reference", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for batch, span in ((1, 128), (1, 256), (4, 256), (8, 256)):
        q, k = torch.randn(2, batch, span, 16, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, span, 32, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(batch, span, 32, device="cuda")
        beta = torch.rand_like(g)
        lengths = torch.tensor([span - 3 - i * 7 for i in range(batch)], device="cuda")
        slots = torch.arange(batch, device="cuda").repeat_interleave(span)
        starts = torch.arange(batch, device="cuda") * span
        boundaries = torch.cat(
            (
                torch.stack((starts, starts + lengths), dim=1).flatten(),
                torch.tensor([batch * span], device="cuda"),
            )
        )
        state_indices = torch.stack(
            (torch.arange(batch, device="cuda"), torch.arange(batch, 2 * batch, device="cuda")),
            dim=1,
        )
        state_indices = state_indices.flatten().to(torch.int32)
        qf, kf, vf = (x.flatten(0, 1).contiguous() for x in (q, k, v))
        alpha = g.exp().flatten(0, 1)
        betaf = beta.flatten(0, 1)
        mask = torch.arange(span, device="cuda")[None, :] < lengths[:, None]
        truth, truth_state = (
            recurrent_fp32(q, k, v, g, beta, lengths) if args.fp32_reference else (None, None)
        )
        for dtype in (torch.float32, torch.bfloat16):
            state = torch.zeros(2 * batch, 32, 128, 128, device="cuda", dtype=dtype)

            def native(q=q, k=k, v=v, g=g, beta=beta, state=state, slots=slots, lengths=lengths):
                state.zero_()
                return prefill(q, k, v, g, beta, state, slots, lengths)

            def flash(
                qf=qf,
                kf=kf,
                vf=vf,
                alpha=alpha,
                betaf=betaf,
                state=state,
                boundaries=boundaries,
                state_indices=state_indices,
            ):
                state.zero_()
                return chunk_gated_delta_rule(
                    # SM100's direct API expects already-normalized Q/K;
                    # its dispatch does not apply use_qk_l2norm_in_kernel.
                    q=l2norm_fwd(qf),
                    k=l2norm_fwd(kf),
                    v=vf,
                    g=alpha,
                    beta=betaf,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=boundaries,
                    use_qk_l2norm_in_kernel=False,
                    output_state=state,
                    state_indices=state_indices,
                    backend="flashinfer",
                )[0]

            expected = native().reshape(batch, span, 32, 128)
            expected_state = state[:batch].clone()
            actual = flash().reshape(batch, span, 32, 128)
            output_error = rrms(actual[mask], expected[mask])
            state_error = rrms(state[:batch], expected_state)
            row = dict(
                batch=batch,
                span=span,
                state_dtype=str(dtype),
                output_rrms=output_error,
                state_rrms=state_error,
                native_output_finite=bool(expected[mask].isfinite().all()),
                flashinfer_output_finite=bool(actual[mask].isfinite().all()),
                native_state_finite=bool(expected_state.isfinite().all()),
                flashinfer_state_finite=bool(state[:batch].isfinite().all()),
                passed=output_error < 0.003 and state_error < 0.003,
            )
            if truth is not None:
                row.update(
                    fla_reference_output_rrms=rrms(expected[mask], truth[mask]),
                    flashinfer_reference_output_rrms=rrms(actual[mask], truth[mask]),
                    fla_reference_state_rrms=rrms(expected_state, truth_state),
                    flashinfer_reference_state_rrms=rrms(state[:batch], truth_state),
                )
                row["reference_qualified"] = all(
                    row[f"flashinfer_reference_{kind}_rrms"]
                    <= min(0.01, 1.1 * row[f"fla_reference_{kind}_rrms"])
                    for kind in ("output", "state")
                )
            if row["passed"] or row.get("reference_qualified"):
                row["fla_us"] = triton.testing.do_bench_cudagraph(native) * 1000
                row["flashinfer_us"] = triton.testing.do_bench_cudagraph(flash) * 1000
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
