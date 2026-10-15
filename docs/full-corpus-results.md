# Full English corpus: corrected feasibility runs

The 2× target is not achieved. The complete three-repetition baseline and
corrected candidate comparison are finished in [full-comparison.md](full-comparison.md).
The candidate points
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
| Synchronous BF16, prefill capacity 16, admission 12 | 4 | 64 | 3470.68 | 187.44 | 421.91 | 0.3932 | Not scored | Not scored |
| Synchronous BF16, adaptive GDN tiles, 192-token bucket | 3 | 64 | 3500.60 | 162.32 | 414.62 | 0.3889 | 1.708 | 64.173 |
| Same, without 192-token bucket | 3 | 64 | 3385.96 | 156.37 | 374.04 | 0.3892 | Not scored | Not scored |
| 192-token bucket + FlashInfer GDN prefill | 3 | 64 | 3465.39 | 159.62 | 413.86 | 0.3863 | 1.867 | 63.934 |
| 192-token bucket + FP16 streaming codec | 3 | 1 | 305.72 | 32.96 | 34.15 | 0.0710 | 1.549 | 64.347 |
| Same FP16 codec | 3 | 64 | 3474.99 | 156.97 | 418.80 | 0.3834 | 1.775 | 64.248 |

The FP32 candidate is frozen at `a6e660c`; tuned-MoE candidates at `e9833d5`.
The GPU differs between some rows, so these values do not establish isolated
A/B effects. Quality coverage is complete for every reported WER/SIM value.

The asynchronous configuration is not selected: its C64 WER is materially
higher than the synchronous candidate. This comparison does not establish
whether the cause is scheduling-dependent sampling or a codec-stream defect.
The capacity-16 run (`0cc53d9`) has little throughput improvement and worse
TTFA; its quality evaluation has not been performed.

The new vLLM baseline `d6ed4f4` has completed all 21 runs on GPU 0:
median C1 throughput 195.67 frames/s and C64 throughput 4510.27 frames/s.
First-repetition quality is C1 WER 1.917%, SIM 64.273; C64 WER 1.842%,
SIM 63.871, each on all 1088 rows. See [the complete baseline table](new-vllm-full-baseline.md).
The frozen candidate `dcd20d3`, with the 192-token bucket and synchronous
codec, completed the same GPU 0 measurement at all seven concurrency
points with three repetitions. Median C1 throughput is 300.81 frames/s and
C64 throughput is 3451.15 frames/s; the throughput target fails at every point.

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
  FLA by about 0.46%, above the original 0.3% cross-backend parity threshold.
  Subsequent independent FP32 recurrence checks found FlashInfer slightly more
  accurate: output RRMS about 0.327% versus FLA 0.336%, and BF16 final-state
  RRMS 0.277% versus 0.307% at B8/T256. This new reference evidence justifies
  an opt-in experiment (`DRAMA3_PREFILL_GDN=flashinfer`), not a relaxation of
  the audio-quality gate. Both partial-batch isolation and exact bucket-boundary
  prompts pass. The full-corpus feasibility at `46fe91b` completed with WER
  1.867%, SIM 63.934 and 3465.39 frames/s at C64. It does not improve the
  observed serving throughput, so the formal candidate retains FLA prefill.
  Earlier non-finite adapter results and failed cross-backend checks are preserved.
- BF16 GDN decode tile tests identified larger tiles with exact output/state
  equality in the tested cases. Subsequent candidates use 16 value rows per
  CTA through B32 and 32 above B32; FP32 retains 8. Serving evidence for that
  subsequent change is included in the frozen `dcd20d3` formal candidate.

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

An MXFP4/BF16 expert diagnostic reduced selected real-input MoE latency from
144/123/102 us to 102/89/79 us at layers 0/19/39. Independent decoding of the
quantized bytes agreed with the FP4 kernel to RRMS 0.0044/0.0042/0.0015, while
the difference from the existing FP8 operator was 0.75/0.16/0.022. This large
precision change is not promoted and has no serving or audio-quality claim.

A separate lossless BF16 transposition/split-K GEMV sweep did not produce a
useful improvement over the existing GEMV on the measured shapes. It is not
integrated. All diagnostic changes remain separate from the frozen `dcd20d3`
formal candidate. That candidate's first C1 repetition scored WER 1.541%,
SIM 64.315 over all 1088 rows. All formal performance repetitions are complete.

Small-batch FP8 expert tile sweeps kept the existing quantization boundaries.
Selected layer output RRMS stayed below 0.000039. Whole B1 autoregressive
decode measured 2.982 ms with default tiles, 2.953 ms with the hot-cache choice,
and 2.941 ms with the cold-cache choice. These are single diagnostics, not
serving gains; the runtime defaults are unchanged.

The FP16 codec experiment (`82ca817`) preserves the original FP32 reference
encoder and makes a separate FP16 streaming decoder. With identical input
codes, streaming waveform RRMS against native FP32 improved from 0.009805
to 0.001557. B1/six-frame graph latency measured 0.888 → 0.736 ms and
B16/six-frame latency 2.224 → 2.059 ms. Exact PCM delivery, slot reuse,
cancellation, bounded draining and error propagation checks pass. Full-corpus
WER/SIM are reported above. C64 throughput did not improve, so this remains
an explicit option rather than a new formal winner. The default codec is BF16.

The baseline C64 audio quality was scored for all three repetitions:
WER 1.842%, 1.917%, 2.093%; SIM 63.871, 63.933, 63.840. These ranges
provide context for stochastic generation; they do not justify dropping a
poor run or inventing a quality acceptance threshold.

Commands, source SHAs, readiness timings and raw request records are under
[evidence/20261015-b300](../evidence/20261015-b300). Full-corpus preparation
encoded 666 unique references in 25.29 s, separately from server startup.
