# Full English corpus: corrected feasibility runs

The 2× target is not achieved. The complete three-repetition baseline and
corrected candidate comparison are still in progress. The candidate points
below are one-repetition feasibility measurements on the entire 1088-row
Seed-TTS English corpus, not a completed cross-runtime speedup claim.

These candidates include the partial-prefill state-isolation fix. The
historical pre-fix results remain available but cannot establish acceptance.
All runs send the same FP32-prepared reference codes, use max 500 frames and
temperature 1 / top-p 0.9 / top-k 30, and keep per-C warmup while skipping the
redundant corpus-wide reference warmup. Every listed point completed 1088/1088
requests. WAVs were scored without dropping high-WER outliers.

| Candidate configuration | GPU | C | Frames/s | TTFA p50 ms | TTFA p95 ms | Mean RTF | WER % | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected FP32 state, packed projection, Triton convolution, admission 6 | 4 | 1 | 296.57 | 34.42 | 35.45 | 0.0733 | 1.901 | 64.207 |
| Same | 4 | 64 | 3256.92 | 172.06 | 404.70 | 0.4089 | 2.177 | 64.019 |
| Same + tuned native prefill MoE | 3 | 64 | 3334.37 | 166.85 | 376.99 | 0.4014 | 1.909 | 63.953 |
| Same + BF16 GDN state | 3 | 64 | 3458.94 | 159.53 | 376.78 | 0.3920 | 2.060 | 63.914 |
| Same BF16 + asynchronous codec and PCM thread | 3 | 1 | 306.14 | 34.53 | 45.92 | 0.0710 | 1.541 | 64.315 |
| Same asynchronous configuration | 3 | 64 | 3547.30 | 156.59 | 373.76 | 0.3758 | 3.835 | 63.626 |
| Synchronous BF16, prefill capacity 16, admission 12 | 4 | 64 | 3470.68 | 187.44 | 421.91 | 0.3932 | Pending | Pending |

The FP32 candidate is frozen at `a6e660c`; tuned-MoE candidates at `e9833d5`.
The GPU differs between some rows, so these values do not establish isolated
A/B effects. Quality coverage is complete for every reported WER/SIM value.

The asynchronous configuration is not selected: its C64 WER is materially
higher than the synchronous candidate. This comparison does not establish
whether the cause is scheduling-dependent sampling or a codec-stream defect.
The capacity-16 run (`0cc53d9`) has little throughput improvement and worse
TTFA; its quality evaluation has not been performed.

The new vLLM baseline `d6ed4f4` has completed its three C1 runs on GPU 0:
median 195.67 frames/s, TTFA p50 46.05 ms, TTFA p95 48.63 ms, mean RTF 0.1107.
Its first C1 repetition scores WER 1.917%, SIM 64.273 on all 1088 rows.
Other concurrency points and the C64 quality reference are pending; these
partial results cannot establish the full acceptance gate.

The native prefill optimization keeps Fish's FP8 quantization and accumulation
math, uses 64-row tiles for at least 1024 input tokens, and atomically reduces
FP32 expert outputs before the BF16 cast. It avoids the eight-way down-output
temporary. Selected real-prompt MoE layers had RRMS at most 0.000014 against
the original path. Real B8/T256 prefill measured 49.77 → 44.64 ms in a same-GPU
isolated diagnostic. End-to-end results above remain the relevant serving evidence.

Additional diagnostics did not justify new serving precision changes:

- Selected large-input INT8/DP4A projections reduced a B1 captured decode
  diagnostic from 2.976 to 2.822 ms, about 5.2%. They add quantization error
  and have no full audio-quality result, so are not enabled in serving.
- FlashInfer GDN prefill requires caller-normalized Q/K on its SM100 dispatch.
  After fixing the diagnostic adapter, outputs were finite but differed from
  FLA by about 0.46%, above the predefined 0.3% parity threshold. This path is
  not integrated. Earlier non-finite adapter results are preserved separately.
- BF16 GDN decode tile tests identified larger tiles with exact output/state
  equality in the tested cases. Subsequent candidates use 16 value rows per
  CTA through B32 and 32 above B32; FP32 retains 8. Serving evidence for that
  subsequent change is pending.

Profiling the corrected synchronous BF16 runtime (`8fdfed2`) over warmup plus
1088 C64 requests recorded 9.18 s in prefill, 18.58 s in decode, and 4.36 s in
codec GPU work. These instrumented timings are not serving performance claims.
Real heterogeneous autoregressive decode (`52bec78`) measured 8.59 ms at B64,
with the two expert GEMMs accounting for roughly 42% of profiled GPU time.
Blocked weight layout measured 8.49 ms and blocked plus autotuning 8.43 ms,
each in a single diagnostic. Generated activations differed between runs;
these are directional observations, not isolated operator A/B results.
The tensor-core W8A16 prototype was slower than the existing BF16 path and
remains diagnostic-only.

Commands, source SHAs, readiness timings and raw request records are under
[evidence/20261015-b300](../evidence/20261015-b300). Full-corpus preparation
encoded 666 unique references in 25.29 s, separately from server startup.
