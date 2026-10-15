# Experiments after the PR #8933 baseline

**Correctness update:** all serving configurations listed below predate
`a6e660c` and contain the partial-prefill GDN state-isolation bug described in
[results.md](results.md). The reproduced failure maps padding primary slots
6/7 in an eight-slot pool to suffix slots 2/3, overwriting unrelated live
requests. The fix reserves padding slots in the primary half and gives them
disjoint suffix slots. Full-model graph checks for 5/6/7 real requests in an
eight-row batch now verify that unrelated convolution, GDN and attention states
are unchanged. Corrected audio and throughput results are pending.

Corrected full-corpus feasibility results are now recorded in
[full-corpus-results.md](full-corpus-results.md). The historical failures below
are confounded by the slot bug and cannot isolate an operator's quality effect.

The complete comparison remains in [results.md](results.md). None of the
experiments below establishes the 2× target. These are one-repetition feasibility
runs on separate B300 devices; they are not controlled three-repetition speedup
claims. Every listed point completed all 192 requests without transport failure.
Audio quality is a separate gate, including all saved samples and high-WER outliers.

| Configuration | GPU | C1 frames/s | C64 frames/s | C64 TTFA p50/p95 ms | C64 mean RTF |
|---|---:|---:|---:|---:|---:|
| BF16 GDN state | 1 | 291.39 | 2389.16 | 175.95 / 635.84 | 0.4757 |
| FlashInfer prefill MoE, FP32 state | 3 | 294.72 | 2182.91 | 171.93 / 1286.63 | 0.5395 |
| Packed shared/router projection, BF16 state | 1 | 296.20 | 2559.56 | 172.86 / 628.40 | 0.4579 |
| Packed projection + Triton prefill convolution, BF16 state | 3 | 299.68 | 2438.52 | 180.13 / 584.70 | 0.4617 |
| Above + admission batch minimum 6 | 4 | — | 2811.70 | 195.08 / 562.33 | 0.4021 |
| Packed projection + Triton convolution + admission 6 + polling async codec, FP32 state | 1 | 298.19 | 2661.32 | 218.18 / 590.99 | 0.4063 |

Source SHAs, flags, CPU affinity, readiness times, commands, raw request timings
and logs are in the corresponding [evidence directories](../evidence/20261015-b300).
Sampling, max output length, reference codes, and client behavior match the 192-row
comparison. GPU identity differs, so do not divide these numbers to claim a gain.

| Configuration | C1 WER % / SIM | C64 WER % / SIM | Provisional quality verdict |
|---|---:|---:|---|
| New vLLM baseline | 2.227 / 62.411 | 1.880 / 62.738 | Reference |
| BF16 GDN state | 2.276 / 61.566 | 2.721 / 61.281 | Within provisional bounds |
| FlashInfer prefill MoE | 1.583 / 61.639 | 4.057 / 61.166 | FAIL |
| Packed projection, BF16 state | 1.286 / 62.893 | 4.008 / 60.891 | FAIL |
| Packed projection + Triton convolution, BF16 state | Pending | Pending | INCONCLUSIVE |
| Above + admission 6 | Not measured | 2.177 / 61.200 | C64 within provisional bounds only |
| Polling async codec, FP32 state | 1.039 / 62.540 | 5.047 / 61.291 | FAIL |

Provisional bounds are baseline WER +1 percentage point and SIM −2 points;
these have not been confirmed by the user. Generation uses stochastic sampling,
and scheduling changes alter the random draws. A failed run is retained and is
not discarded as noise. The full 1088-row English corpus is being evaluated to
reduce dependence on the small subset; no full-corpus verdict is available yet.

## Isolated diagnostics

- Packed projection combines shared-expert gate/up, router, and shared gate
  without quantization. Tested full-MoE output RRMS was at most 0.000385 across
  selected layers and batch sizes. This does not supersede the audio gate.
- Triton prefill convolution retained exact BF16 output and recurrent convolution
  state in the tested shapes. At B8/T256 the operator measured 320.35 → 34.43 µs;
  real eight-prompt prefill measured 58.66 → 50.18 ms. These are diagnostics.
- BF16 GDN state reduced a B64 decode diagnostic from 6.930 to 6.484 ms.
  Default state remains FP32. Blocked FP8 weights and autotuning did not show
  a meaningful decode improvement in their diagnostics.
- FlashInfer prefill MoE reduced isolated real prefill time but regressed the
  serving feasibility run and failed its audio gate. The cause of the serving
  regression is not established; native prefill remains the default.
- Polling async codec passed byte-for-byte PCM, slot reuse, bounded queue,
  cancellation, and completion-order checks. C1 TTFA nevertheless rose to
  45.90 ms because completed audio waited for the model loop. An independent
  delivery thread also passed PCM/state checks; its serving result is pending.
- Fusing BF16 gating into GEMV preserved exact outputs in the diagnostic.
  The 512-input shared projection improved 2.53 → 2.00 µs and attention gating
  5.71 → 4.56 µs, but the 6144-input tower projection regressed 4.76 → 6.12 µs.
  This prototype is not enabled in serving.
- Dense FP8 failed numerical/performance diagnostics and was removed from serving.
  Rejected operator variants and the profiler's unsupported nested-capture failure
  remain recorded; they are not benchmark wins.

## Experimental switches

All switches below default to the conservative path and are recorded by the
launcher. They are experiments, not recommended production settings.

| Switch | Default | Experiment |
|---|---|---|
| `DRAMA3_GDN_STATE_DTYPE` | `float32` | `bfloat16` recurrent state |
| `DRAMA3_PACKED_MOE_PROJECTION` | off | `1`: packed shared/router projection |
| `DRAMA3_PREFILL_CONV` | `torch` | `triton`: causal convolution with exact end state |
| `DRAMA3_PREFILL_MOE` | `native` | `flashinfer`: failed audio gate |
| `DRAMA3_PREFILL_MOE_TUNED` | off | `1`: 64-row native tiles and FP32 atomic reduction for large prefills |
| `DRAMA3_MOE_LAYOUT` | `major_k` | `blocked`: unchanged FP8 bytes, different layout |
| `DRAMA3_AUTOTUNE_CACHE` | unset | FlashInfer tactic cache path |
| `DRAMA3_ADMISSION_MIN_BATCH` | `1` | Bounded wait for a larger prefill batch |
| `DRAMA3_ADMISSION_MIN_RUNNING` | `32` | Minimum active requests before admission waits |
| `DRAMA3_ADMISSION_MAX_WAIT_MS` | `100` | Maximum admission delay |
| `DRAMA3_PREFILL_MAX_BATCH` | `8` | `16`: larger short-prompt batches with separate padding state |
| `DRAMA3_ASYNC_CODEC` | off | `1`: bounded codec CUDA stream |
| `DRAMA3_AUDIO_DELIVERY_THREAD` | off | `1`: event-waiting PCM sender, requires async codec |

The full 1088-row run uses `--skip-reference-warmup` on both runtimes: identical
pre-encoded references need no additional corpus-wide preparation pass. Per-C
warmup remains `max(8, concurrency)` requests. This is a separate protocol from
the historical 192-row comparison and will be reported separately.
