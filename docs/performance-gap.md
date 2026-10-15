# Why the 2× target remains open

The new vLLM-Omni baseline reaches 4510.27 frames/s at C64. The target is
9020.54 frames/s. The corrected standalone feasibility runs are around
3500 frames/s; they need about 2.6× their present throughput. Small codec or
tile changes do not bridge this gap.

The heterogeneous B64 decode diagnostic at `52bec78` measures 8.59 ms for one
autoregressive frame per request, after real voice-cloning prompts and previous
generated frames. Even an ideal stream consisting entirely of those full-batch
decode steps would provide only about 7452 frames/s, before prefill, codec,
scheduling, transport or partially occupied batches. This is an extrapolation
from the diagnostic, not a measured serving throughput or a universal hardware
limit. At C64 the target allows only 7.095 ms per full-batch step including all
amortized work.

The profiler attributes about 42% of decode GPU time to the two expert GEMMs,
7.5% to recurrent GDN updates, and substantial remaining time to dense
projections and their reductions. A separate synchronized C64 stage diagnostic
records 9.18 s of prefill, 18.58 s of decode and 4.36 s of codec GPU work over
warmup plus the full corpus. The diagnostic synchronizations affect execution;
these stage totals must not be substituted for serving measurements.

Measured experiments constrain the next implementation choices:

| Experiment | Evidence | Decision |
|---|---|---|
| Native prefill MoE tiling | Real B8/T256 prefill 49.77 → 44.64 ms; small numerical difference | Included in formal candidate |
| 192-token prefill bucket | Avoids padding most corpus prompts to 256 tokens; complete audio quality | Included in formal candidate |
| BF16 GDN state and larger decode tiles | Tested recurrence and state isolation; matches upstream state storage | Included in formal candidate |
| FlashInfer GDN prefill | Faster isolated operator, no end-to-end C64 gain | Optional, not selected |
| FP16 codec | Closer to FP32 waveform reference, faster short chunks, no C64 gain | Optional, not selected |
| Small FP8 expert tile changes | Roughly 1.4% in one B1 decode diagnostic | Diagnostic only |
| MXFP4 experts | Faster selected MoE layers, large difference from checkpoint FP8 math | Rejected for serving |
| Async codec | Higher C64 WER in the full-corpus feasibility run | Not selected; cause not isolated |

The next substantial experiment should address expert weight traffic and
prefill/decode GPU work together. Candidate directions are grouped expert
kernels specialized for the actual token-to-expert occupancy, and execution
that combines new prompt tokens with ongoing decode work. These are unimplemented
hypotheses, not promised speedups. They need real heterogeneous activations,
independent numerical checks, prefill state-isolation checks, complete WER/SIM,
and the same three-repetition serving comparison before promotion.

The existing profiles do not establish a CPU scheduling bottleneck large enough
to justify moving the engine to Rust. A language rewrite alone is not a measured
path to the target.

See [the full-corpus experiments](full-corpus-results.md) and
[raw profiler evidence](../evidence/20261015-b300/heterogeneous-decode-b64).
