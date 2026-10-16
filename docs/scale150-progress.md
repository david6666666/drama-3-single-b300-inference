Latest update: full resident MoE and M128 global-buffer implementations pass
native arithmetic checks, but both fail their declared local performance gate.
Resident fusion is14–37% slower in the three real routing cases; M128 reduces
latency by only0.57–2.87%, below the required5% per real case. Neither is promoted
to the runtime. The previous occupancy inference from source register hints is
withdrawn after disassembly; actual machine code has different resource policies.
The best complete-quality C64 observation remains4726.771 frames/s and the target
remains6765.407. Full results and preserved failures are in the
[evidence](../evidence/20261016-ceiling/README.md).

A subsequent PDL-enabled M128 candidate passes1498 exact checks but misses
the continuation threshold in two of three real cases in its first timing round.
Pressure-case timing, three repetitions and service integration are not run.
The next metadata/consumer-wait schedule remains under separate validation.

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

## Latest C64 investigation

The best complete-quality single observation is now 4726.771 frames/s on
`e7cda1e`, with TTFA p50/p95 145.645/273.682 ms, mean RTF 0.288130,
WER 1.599263% and SIM 64.412759. It is the cache-off control of one native
pipeline/deferred-admission feasibility pair, not a repeated winner. The
cache-on arm scores all outputs too but observes 4585.126 frames/s; it stays
experimental. All-concurrency three-repeat acceptance is still incomplete.

A current nonblocking timeline and fixed-work replay explain why codec-local
speedups do not close the service gap. Native model execution remains the
main budget; new resident-expert work is an unqualified operator prototype.
See [complete results, failures and limitations](../evidence/20261016-ceiling/README.md).

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
totals explain hypotheses rather than predict an end-to-end speedup. That
historical synchronized probe does not identify steady-state occupancy.
The newer continuous-admission traces show mean active decode batches of 60.718
(upstream) and 61.071 (standalone mixed), both close to C64; their different
profiling conditions prevent a controlled speed comparison. Current work targets
mixed prompt span, shared model execution and codec overlap,
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
| Native input fusion with synchronous codec | GPU0 same-source C64 control/fused: 3596.20 / 3643.99 frames/s; WER 1.608% / 1.742%, SIM 64.155 / 64.115; all 1088 outputs in each arm scored | Both single groups pass quality; small feasibility gain only, far below the throughput target |
| Larger native FP8 expert column tiles | All 36 tested configurations preserve original native output exactly; wider tiles lose to the original N128 configuration, itself slower than current FlashInfer on these inputs | No serving dispatch change; original-native/current-FlashInfer differences are recorded separately |
| Mixed backbone layer/graph/state validation | Expanded D58/P6, D40/P3 and D12/P1 cases pass 2190 checks, including reused slots and untouched-state equality | Numerical/state evidence only; GPU0 timing unqualified after an external process appeared; [checks](../evidence/20261016-mixed/results/mixed-layers-expanded-v2.json) |
| Upstream C64 continuous-admission profile | 256 requests: 802 CPU scheduler steps, not piecewise fragments; admission interval has 36 mixed and 284 decode steps, mean decode active 60.718 | Initial fill and final drain retained; not formal throughput or SM utilization; [summary](../evidence/20261016-mixed/results/vllm-continuous-schedule.json) |
| Actual-input native FP8 expert counters | Six FC1/FC2 launches at layers 0/19/39: DRAM 4.320–5.792 TB/s, tensor-pipe active 3.46–4.48%; all original top-8 weights/scales retained | Consistent with weight movement limiting these sampled kernels; not an all-layer mean or serving speedup; [counters and limits](../evidence/20261016-mixed/results/ncu/summary.json) |
| Mixed backbone, synchronous codec, complete same-source pair | GPU6 C64 control/mixed: 3816.47 / 4094.18 frames/s, WER 1.608% / 1.583%, SIM 64.489 / 64.286; 1088 outputs scored per arm | Both single groups pass quality; observed +7.28% is confounded by output-length/tail differences and remains below baseline; [control](../evidence/20261016-mixed/results/mixed-control-gpu6/benchmark/summary.json), [mixed](../evidence/20261016-mixed/results/mixed-candidate-gpu6/benchmark/summary.json) |
| Mixed continuous-service event timeline | GPU2, 384 requests: 9.998 s window; pure decode 6.060 s, mixed plus teacher tower 1.969 s, synchronous codec 1.454 s | All events on stream 0; CPU first-frame wait overlaps prior GPU work; no causal speedup attribution; [phases](../evidence/20261016-mixed/results/mixed-timeline-phase-analysis.json) |
| Reduced mixed prompt span | Three shapes / 339 exact output-and-state checks pass; 512 → 256/192 span saves 2.53–2.62 ms per refill in the isolated diagnostic | Preserve packed row capacity; long prompts require fallback; no end-to-end claim from these timings; [checks](../evidence/20261016-mixed/results/mixed-span.json) |
| Mixed span 256 with fallback 512, complete service pair | GPU0 C64 same-source control/candidate: 3997.02 / 4082.76 frames/s; TTFA p95 263.725 / 276.424 ms; WER 1.951% / 1.834%, SIM 63.752 / 63.624, all 1088 per arm scored | Both single groups pass quality; observed +2.15% with worse p95 remains below baseline; do not conflate with the GPU6 mixed pair; [control](../evidence/20261016-mixed/results/span-control-gpu0/benchmark/summary.json), [candidate](../evidence/20261016-mixed/results/span-candidate-gpu0/benchmark/summary.json) |

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

That synchronous-codec pair at `3a45588` has now completed on the frozen physical
GPU0. The control's TTFA p50/p95 is 150.844/345.124 ms and mean RTF 0.367997;
the fused input path is 142.002/344.212 ms and 0.360835. All generated outputs
were scored, including outliers. This is one feasibility repetition per arm,
not a completed all-concurrency performance/quality campaign. The 3643.99
frames/s result remains below the C64 baseline of 4510.27 and target of 6765.41.
The fusion is opt-in; no winning default or successful README example is nominated.

A separate 384-request event timeline at `1a88b79` excludes warmup using raw
request timestamps. Mean active decode batch is 49.65; prefill event spans sum
to 2.436 s, decode to 6.612 s, and codec/copy to 2.308 s. Codec spans overlap
decode by 1.935 s. These spans can include submission gaps and are not SM
utilization or summed kernel durations. Codec already overlaps substantial
decode work, so its standalone duration is not an available service speedup.
The instrumented run is diagnostic only and is excluded from acceptance.

## Mixed scheduling: complete feasibility, target still unmet

The latest mixed-backbone pair uses the same runtime `a25c68d`, GPU6 and full
1088-request corpus in each arm. Only `DRAMA3_MIXED_BACKBONE` changes; both arms
use FlashInfer prefill MoE, FLA prefill GDN, BF16 recurrent state, native fused
MoE inputs and synchronous BF16 codec. The frozen upstream deployment instead
uses asynchronous FP16 codec and MPS; these are recorded differences, not an
isolated estimate of their performance contribution.

| Metric | Separate-backbone control | Mixed candidate |
|---|---:|---:|
| Frames/s | 3816.466 | 4094.177 |
| TTFA p50 / p95, ms | 140.674 / 291.234 | 147.565 / 267.444 |
| Mean RTF | 0.346050 | 0.331417 |
| Corpus WER, % | 1.607636 | 1.582517 |
| Mean SIM | 64.48894 | 64.28628 |
| Successful / WER scored / SIM scored | 1088 / 1088 / 1088 | 1088 / 1088 / 1088 |
| Actual acoustic frames | 102946 | 101162 |
| Makespan, s | 26.9742 | 24.7088 |
| Last dispatch to final completion, s | 1.9016 | 1.1362 |

Both individual quality groups pass the unchanged C64 bounds, with no excluded
outliers: [control quality](../evidence/20261016-mixed/results/mixed-control-quality.json.gz)
and [candidate quality](../evidence/20261016-mixed/results/mixed-candidate-quality.json.gz).
The observed +7.28% throughput difference is not a repeated fixed-workload
speedup. The candidate generated 1.733% fewer frames, and 500-frame outputs fell
from four to one. Final drain shortened by 0.7653 s, about 34% of the makespan
difference. All requests and tail remain included. Sampling counters, row order
and graph preparation can change the random trajectory; no single cause of
output differences has been established. The candidate is only 90.77% of the
4510.27 frames/s C64 baseline and needs another 65.24% to reach 6765.41.

The separate 384-request mixed diagnostic excludes warmup and retains all
requests. Its 8.027 s continuous-admission interval includes initial fill; mean
pure-decode batch is 61.071 there, versus 14.025 during the 1.972 s final drain.
Thus global mean batch 45.367 does not describe sustained occupancy. The
`mixed_backbone` event includes the teacher-forced prefill tower. CPU first-frame
wait of 2.031 s mostly waits for earlier GPU work and is not independent CPU
computation. Synchronous codec occupies 1.454 s of the 9.998 s event window;
perfectly hiding it would give only an ideal 1.170× budget, not a predicted gain.
The old-request tower is just 100.93 ms, about 1.01% of the window, so fully
unifying its labels and sampling has lower priority than prompt span and safe
codec scheduling. See the [source-grounded analysis](high-concurrency-analysis.md)
for counters, source references and interpretation limits.

The next span experiment uses `9247e4b` on physical GPU0, with both arms already
using mixed backbone. Control / span-256-with-512-fallback has TTFA p50
146.182 / 147.932 ms, p95 263.725 / 276.424 ms and mean RTF 0.329850 / 0.323508.
Its 3997.02 / 4082.76 frames/s is another single feasibility comparison, still
below baseline. Both groups now have complete WER/SIM for all 1088 outputs:
control WER 1.950934%, SIM 63.75201; candidate WER 1.833710%, SIM 63.62432.
Both pass the unchanged single-group quality bounds, with all outliers included:
[control quality](../evidence/20261016-mixed/results/span-control-quality.json.gz),
[candidate quality](../evidence/20261016-mixed/results/span-candidate-quality.json.gz).
This result must not be multiplied by the separate GPU6 difference or described
as a validated all-concurrency improvement. The earlier asynchronous
codec failures remain unresolved despite the passing synchronous mixed pair.

Raw compact evidence and experiment plans are under
[20261015-scale150](../evidence/20261015-scale150) and
[20261016-mixed](../evidence/20261016-mixed).
Large profiler traces and audio stay in the recorded local artifact directories.
OmniRSI PR #9 has its conflicts resolved and a source-grounded concurrency
knowledge record, but no successful optimization claim.


The later fixed dynamic codec replay reproduces a default-capture divergence
with identical request ordering, while a dedicated codec capture stream retains
exact codes, PCM and all state across sync/poll/thread modes. Actual ATen BLAS
workspace ranges are shared in the former and disjoint in the latter. See
[the controlled diagnostic](../evidence/20261016-mixed/results/codec-dynamic/summary.json).
Its additional host reads affect overlap timing; the historical async quality
failures remain recorded pending a fresh complete service comparison. The
experimental flag defaults off.

[Actual mixed MoE counters](../evidence/20261016-mixed/results/ncu/mixed/summary.json)
also show near-one-footprint expert-weight reads for the three sampled layers.
They reject a large redundant-HBM-read hypothesis for these inputs, without
establishing an all-layer lower bound or a service speedup.


The fresh GPU0 workspace pair is complete (`7618212`, all 1088 outputs per arm).
Shared capture: 4307.662 frames/s, WER 6.162606%, SIM 63.124403 (**quality fail**).
Private capture: 4360.917 frames/s, TTFA p50/p95 143.351/278.698 ms, mean RTF
0.307843, WER 1.590890%, SIM 64.191350 (**single-group quality pass**).
This supports the reproduced correctness fix, not a repeated speedup claim or
qualification of every historical async failure. New service startup rejects
async codec without `DRAMA3_CODEC_DEDICATED_CAPTURE_STREAM=1`; synchronous defaults
are unchanged. C64 remains below both the 4510.27 baseline and 6765.41 target.
[Full pair and raw quality](../evidence/20261016-mixed/results/service-feasibility-summary.json).


## Documentation after acceptance

After the full target and quality matrix pass, update the main README with a
source-accurate architecture diagram, the retained runtime optimizations and
standalone/vLLM-Omni/SGLang-Omni results at every tested concurrency. Report
throughput, TTFA p50/p95 and mean RTF together with exact revisions, precision,
workload and complete quality coverage. Historical or single-run observations
must stay distinguished from the final repeated comparison. The architecture
must show the actual Python engine and direct GPU operators; Rust is not yet an
implemented component. This requested publication remains pending acceptance.
