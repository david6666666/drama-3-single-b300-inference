# Native C64 ceiling investigation (2026-10-16 Asia/Shanghai)

The target remains **6765.407263 frames/s**, with full native top-8 computation,
BF16 dense/GDN, original expert FP8/scales, all 1088 outputs, and the frozen
WER/SIM gates. **Not achieved.** These are diagnostic and one-run feasibility
results, not the required all-concurrency three-repeat acceptance matrix.

The isolated B64 measurement (8.275183868 ms, approximately 7734 frames/s) is an
achieved full-batch decode reference, **not a hardware upper bound or a service
lower bound**. Latest full asynchronous profiling attributes the reference gap
to decode inefficiency/underfilling/contention and required mixed prompt work;
client dispatch starvation is not supported. See
[the complete accounting](results/theory-vs-service.md) and
[rich service timeline ledger](results/rich-timeline-service-ledger.json).

| Same-source feasibility arm | GPU | Frames/s | TTFA p50/p95 ms | RTF mean | WER % | SIM |
|---|---:|---:|---:|---:|---:|---:|
| Deferred-submit control | 6 | 4366.914 | 142.303 / 278.628 | 0.306936 | 1.515532 | 64.275579 |
| Deferred-submit candidate | 6 | 4430.811 | 151.637 / 283.801 | 0.306791 | 2.043038 | 63.763861 |
| Compact/layout/conv control | 0 | 4371.393 | 144.441 / 279.874 | 0.308716 | 1.599263 | 64.354878 |
| Compact/layout/conv candidate | 0 | 4306.693 | 143.598 / 271.163 | 0.302134 | 1.875576 | 63.868203 |
| Short-padding control | 7 | 4230.068 | 147.179 / 278.054 | 0.313088 | 1.850456 | 63.784350 |
| Short-padding candidate | 7 | 4362.323 | 142.507 / 274.634 | 0.301702 | 1.775098 | 64.073121 |
| Four-change control | 0 | 4353.910 | 142.503 / 279.864 | 0.305447 | 1.758352 | 64.140282 |
| Four-change candidate | 0 | 4591.038 | 140.774 / 268.097 | 0.295911 | 1.925814 | 63.693694 |

All eight arms completed and scored 1088/1088 outputs without filtering. They pass
the frozen C64 quality limits (WER≤2.950934%, SIM≥61.881589). Deferred submission produces2.677% more frames
but takes1.197% longer. The combined operator arm produces2.106% more frames and
has1.092s longer drain; its lower complete throughput remains a failure, despite
improved TTFA p95/RTF and faster fixed-state mixed graphs. Do not remove the tail
or substitute intake-only throughput for the acceptance score.

Short padding alone observes +3.127% frames/s, −3.512% makespan and −0.496%
output frames in its one same-source pair. Combining compact GDN, mixed layout,
empty convolution guard and short padding observes +5.446% frames/s, with
+0.858% output frames. The latter is only 1.0179× the frozen vLLM baseline,
not 1.5×. These are feasibility observations, not repeated speedup claims.
See [short-padding](results/short-padding-service-summary.json) and
[combined full-service evidence](results/combined-native-service-summary.json).

Implemented candidates remain opt-in:

- `DRAMA3_COMPACT_GDN=1`: compact static chunk descriptors and skip invalid output/recompute tiles;2099 exact checks in6 cases, including30 real GDN layers.
- `DRAMA3_FUSED_MIXED_LAYOUT=1`: consume separate decode/prefill buffers and strided gates without intermediate copies;339 exact checks.
- `DRAMA3_PACKED_CONV_GUARD=1`: skip convolution tiles with no valid output/state writes;339 exact checks. The layout+conv combination passed another339 checks.
- `DRAMA3_DEFER_CODEC_SUBMIT=1`: submit prior audio after the next actual model launch, retaining its original producer event;414-event dynamic replay preserves codes,PCM,and states exactly. The service feasibility above does not justify promotion.
- `DRAMA3_CODEC_SPLIT_POLICY=min_launches`: CPU legal-partition checks pass; the initial dynamic bit-exact gate fails and the option is not promoted. See raw evidence before using it.
- `DRAMA3_SHORT_PADDING_CONTEXT=1`: only explicitly reserved decode rows use short attention context. All 1023 exact checks across 9 cases pass; both full service quality arms pass as reported above. The different local 7.23ms B64 diagnostic used different input/state from the old 8.275ms reference and is not a new universal ceiling.

Current rich timeline records197 mixed calls,183228 actual prompt tokens in208384
captured rows (13.73% padding), and2265 codec calls from2122 frame-bucket groups.
Only143 calls (6.31%) are avoidable same-bucket DP splits;833 additional groups
come from differing frame buckets within an emit. These counts are not time
saving percentages.

A fixed-state interference probe passed all60 code/PCM/state comparisons. Four
B64 steps alone take33.33ms; adding three real8-frame codec jobs takes34.65ms
(B1),35.57ms(B4),or36.92ms(B8). The concurrent pair still finishes earlier than
serial execution. Model-high stream priority does not materially improve the
pair. The actual graph flags were queried (1, no UseNodePriority); the priority
hypothesis was tested rather than assumed. See
[the controlled probe](results/codec-priority/summary.json).

Existing native FlashInfer GDN was reconsidered on the current mixed shape.
The original numerical gate failed on a private padding sequence. A separate
FP32-reference/causal-isolation diagnostic preserves that failure, verifies the
real same-input per-layer errors and isolates padding, but observes final
prefill hidden RRMS0.09118. Its local timing gain is only1.776%, below the
predeclared5% threshold, so no new full service run or promotion follows.

A separate numeric analysis of the codec partition failure finds all real codec
slots and all model state exact; differing final state belongs only to the private
scratch slot. Five of80 PCM outputs differ (RRMS0.317–0.572%, SNR44.86–49.97dB).
This does not pass the original exact gate or establish WER/SIM; it remains an
unpromoted numerical investigation, with every original failure retained.

A separate default-off native MoE residual fusion failed its first static exact
check. The original failure is retained; no graph speed or service result is
claimed for it. Native codec SM limiting is also under investigation: tiny
GreenContext captures obey their SM cap, but replaying a graph captured in the
primary context does not. Whole DAC capture has hit cuDNN initialization errors;
there is no qualified resource-cap performance result yet.

Source SHAs, experiment hypotheses, commands, raw per-request summaries,
complete quality records and failures are indexed by [files.json](results/files.json).
Gzip files retain the original uncompressed SHA256. Full WAVs and large tensor
captures remain under `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016`;
they are not duplicated into Git. No successful README example is added.
