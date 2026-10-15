# Current concurrency optimization campaign

Status: **INCONCLUSIVE; the all-concurrency target has not been achieved.**
The latest user requirement supersedes the earlier 2× target: each concurrency
must achieve at least 1.5× vLLM-Omni `d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`.
WER may rise by at most one percentage point and SIM may fall by at most two
points, compared with the same concurrency. Both quality bounds are confirmed.
The [acceptance issue](https://github.com/ThinkFlowLab/OmniRSI/issues/10),
[frozen plan](../evidence/20261015-scale150/plan/plan.json) and
[validator](../evidence/20261015-scale150/validator/validate.py) require 1088
requests per repetition, three repetitions at each of seven concurrency levels,
complete quality coverage and matching source, hardware and workload controls.
No README success example will be added to OmniRSI before every gate passes.

The user also confirmed that further optimization must preserve native model
computation: retain all eight routed experts and introduce no additional lossy
quantization. Existing precision-reduction diagnostics remain historical
rejections. Operator layout, fusion and scheduling experiments must still pass
the unchanged performance and audio-quality gates.

## Complete baseline quality

All 21 vLLM performance groups now have complete WER/SIM scoring: 22,848 saved
waveforms, with no discarded outliers. WER aggregates the word-error counts
across all three repetitions; SIM is the mean of their per-sample scores.

| Concurrency | Required frames/s | vLLM WER % | vLLM SIM |
|---:|---:|---:|---:|
| 1 | 293.500 | 1.91465 | 64.05233 |
| 2 | 518.567 | 1.87278 | 64.05586 |
| 4 | 946.818 | 1.90349 | 64.07027 |
| 8 | 1652.346 | 1.73881 | 64.16508 |
| 16 | 2764.037 | 1.84487 | 63.92698 |
| 32 | 4296.985 | 1.85604 | 64.14505 |
| 64 | 6765.407 | 1.95093 | 63.88159 |

The validator independently rechecked the raw performance rows and all 21
quality groups. The new files supplement the nine historical quality groups;
they do not replace their outputs. See
[baseline completeness](../evidence/20261015-scale150/results/baseline-completeness.json)
and [quality aggregates](../evidence/20261015-scale150/results/vllm-quality-all-repetitions-summary.json).
No new candidate has complete evidence for all acceptance gates.

## Why scaling needs more than a faster decode kernel

An unchanged upstream service trace gives a median GPU span of 9.458 ms for
24 pure B64 decode graphs. The standalone heterogeneous B64 diagnostic is
8.589 ms, despite lower end-to-end throughput. These are diagnostic observations,
not a controlled serving speedup claim. Upstream kernel durations overlap and
their sum exceeds elapsed graph time. Comparing a mixed-batch average with a
full B64 operator would also misattribute the difference.

The standalone stage diagnostic spends substantial time outside decode:
9.18 s in prefill, 18.58 s in decode, and 4.36 s in codec, with mean decode
occupancy of 51.5 requests. Its synchronization changes execution, so these
totals explain hypotheses rather than predict an end-to-end speedup. Current
work targets padded prompt computation, admission occupancy and codec overlap,
while retaining numerical and state checks. A Rust rewrite is not yet supported
by evidence of a dominant CPU cost.

## Experiments and limitations

Each result below is a feasibility run or diagnostic, not a promoted three-run
performance claim. The unchanged formal candidate remains in
[the full comparison](full-comparison.md).

| Experiment | Observation | Decision |
|---|---|---|
| FlashInfer prefill MoE with corrected state isolation | C64 3527.70 frames/s; WER 1.817%, SIM 64.122; all 1088 outputs scored | Quality looks acceptable for this run; throughput remains far below 6765.41 |
| Six-request prefill bucket | State checks pass; C64 3466.27 frames/s | No substantial serving gain |
| Small MoE hot tiles | B8 pure decode 5.245 → 4.925 ms; layer tile errors within the diagnostic bound | Continue B2/B4 isolation before serving integration |
| Native small MoE versus FlashInfer at B8 | One real layer RRMS 0.03476 exceeds predeclared 0.03 | Do not change the dispatch default |
| MXFP8 expert re-encoding | Layer 0 RRMS 0.23045 versus native; independent implementation parity 0.01071 | Rejected by the predeclared 0.1 error limit; no service test |
| Packed variable-length attention primitive | CUDA Graph, changing lengths, empty sequences and causal GQA checks pass | Proceed only to isolated full-prefill validation |
| Complete packed prefill versus independent single requests | Worst state RRMS 0.322; unchanged padded control also reaches 0.394 | Both exceed 0.03; retain failed diagnostics and isolate each operation on identical inputs |
| Packed prefill with ordered native reduction | All 4546 expanded layer/graph/state checks pass at B2/3/5/6/8; C64 3645.87 frames/s versus same-configuration padded 3383.91 | Single-run feasibility improvement; below both vLLM and target |
| Packed prefill audio quality | All 1088 scored; WER 1.624%, SIM 64.364 | Within confirmed bounds for this run; all-concurrency repeated acceptance still missing |
| Fused small-batch tower projection | All 144 operator checks pass, maximum RRMS 0.000393; real B4 decode 4.190 → 4.059 ms; C4 serving 725.92 versus fresh unchanged 756.05 frames/s | Not selected: microbench gain did not improve service; complete WER 1.348%, SIM 64.067 |
| Batched multi-step decode | 24 full/partial-batch checks pass exactly, including limits and context boundaries; C4 740.25 frames/s, TTFA p95 74.267 ms | Not selected: lower throughput and higher first-audio latency than fresh control; complete WER 1.859%, SIM 64.181 |
| Concurrent GEMM graphs | Separate and shared capture streams both match serial outputs in the tested primitive | No workspace race demonstrated; do not attribute prior quality problems to this hypothesis |
| Packed prefill/decode overlap | GPU0: serial 80.656 ms, overlap 75.123 ms for one prefill plus five decode steps; hidden and all states exact | Isolated opportunity only; first-frame sampling, scheduling and codec excluded; no service integration |
| Native shared/routed expert overlap | GPU6: B64 8.452 → 8.418 ms and B4 4.195 → 4.134 ms per decode step; ten steps preserve codes and all state exactly | Small diagnostic difference only; no serving improvement established |
| Native BF16 projection tiles | All 144 tested tiles pass on real B64 projection inputs, max RRMS 0.000146; none beats the existing operator | Keep existing projection dispatch; no new quantization or serving default |
| Real decode/codec overlap replay | Sync, polled and threaded delivery produce exact PCM and model/codec state for two generations of 64 requests, including slot reuse | Isolates stream safety under fixed request ordering; does not resolve earlier service-quality failure or establish throughput |
| Packed prefill plus native asynchronous codec | GPU6, one C64 full-corpus trial per arm: sync 3657.93 frames/s, async 3729.52; both score all 1088 outputs | Below target; a 2% feasibility difference is not a repeated serving claim |
| Fused native MoE input preparation | All 64 buffer cases exact; B64 ten-step decode 8.456 → 8.275 ms with exact codes and all mutable state | Retains original FP8 values, scales and all 256 router logits; no service gain established yet |
| Disable native MoE programmatic dependent launch | Exact codes and state; B64 8.448 → 8.529 ms | Slower in this diagnostic; retain the existing launch policy |
| Native input fusion with async codec, fresh same-source control | GPU6: control 3751.27 frames/s, WER 8.047%, SIM 62.623; fused 3860.94 frames/s, WER 3.266%, SIM 63.716; all outputs scored | Both fail the C64 WER bound; reject this configuration despite exact operator/replay checks |
| Packed prefill/codec concurrent replay | B1/F6, B4/F16 and B8/F64 codec work overlaps six real prompt prefills; all model/codec state, hidden output and PCM exact | No data race reproduced in these cases; dynamic service quality remains unresolved |

The original packed experiment was stopped at its numerical gate. A separately
declared layer-isolation experiment retains the 0.03 operator threshold and
exact state-ownership checks; it does not treat whole-network drift as a pass.
Neither graph parity nor layer parity replaces full audio quality. The frozen
acceptance plan and validator have not been relaxed.

The layer-isolation probe initially also found a graph/eager mismatch at B5
when the existing atomic FP32 reduction was enabled. Disabling the tuned
atomic prefill path removed that mismatch in all tested cases, without changing
the numeric threshold. Experimental packed serving therefore rejects the atomic
configuration. Both arms of its service comparison use the untuned native
reduction. Their TTFA p50/p95 are respectively 145.818/347.136 ms (packed) and
164.584/432.392 ms (padded); mean RTF is 0.364525 and 0.391865. These are single
feasibility measurements, not a repeated A/B claim. The raw timings, launch
commands, partial failures and complete packed quality remain in the evidence.

The first overlap diagnostic used GPU1. External vLLM workers were observed on
that device at completion, so its timing is explicitly unqualified. The GPU0
retry above used an empty device; those external processes were left alone.
The packed graph validator was also strengthened to snapshot untouched slots
before graph replay, in addition to the existing eager-path isolation check.

Full top-8 routing instrumentation observed a mean of 161.75 active experts per
B64 layer in one real decode snapshot. Padding each expert to eight rows would
give mean row utilization of 37.4%; this describes the routing distribution,
not measured hardware utilization. All experts are retained. An external job
was observed on GPU0 after that capture, so its timing is unqualified. Subsequent
stream and projection diagnostics use an idle GPU6; formal acceptance still
requires the frozen physical GPU0. Other users' GPU processes are left intact.

The packed codec pair uses the same frozen runtime `66dd6fc` on GPU6. Sync
TTFA p50/p95 is 149.145/346.882 ms, mean RTF 0.366906, WER 1.767% and SIM
64.035. Async TTFA is 146.331/345.504 ms, mean RTF 0.351421, WER 2.462% and
SIM 63.924. Both individual groups meet the confirmed C64 quality bounds.
This does not erase the earlier async quality failure on a different serving
configuration, establish its cause, or satisfy the repeated GPU0 campaign.
The subsequent same-source native-fusion trial fails quality in both arms:
the control has WER 8.047% and the fusion arm 3.266%, above the C64 limit of
2.951%. These failures supersede any suggestion that the asynchronous serving
configuration is ready for selection. They are retained without outlier
filtering or retries for a favorable sample. Exact decode/codec and
prefill/codec replay checks did not reproduce a race; neither establishes the
cause of the service failures. Synchronous codec isolates the next fusion test.

A separate 384-request event timeline at `1a88b79` excludes warmup using raw
request timestamps. Mean active decode batch is 49.65; prefill event spans sum
to 2.436 s, decode to 6.612 s, and codec/copy to 2.308 s. Codec spans overlap
decode by 1.935 s. These spans can include submission gaps and are not SM
utilization or summed kernel durations. Codec already overlaps substantial
decode work, so its standalone duration is not an available service speedup.
The instrumented run is diagnostic only and is excluded from acceptance.

Raw compact evidence and experiment plans are under
[20261015-scale150](../evidence/20261015-scale150).
Large profiler traces and audio stay in the recorded local artifact directories.
OmniRSI PR #9 has its conflicts resolved and a source-grounded concurrency
knowledge record, but no successful optimization claim.
