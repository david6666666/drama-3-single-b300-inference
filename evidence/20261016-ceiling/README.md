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

The original MoE residual fusion failed its first exact check. The cause was
native BF16 FMA contraction: the reference compiler fused shared×gate+routed,
while the first candidate used separate rounded operations. Commit `6ff8d80`
restores that native FMA explicitly and passes 2374 exact checks, including all
40 live boundaries, full model/request/RNG state and shared-stream compatibility.
Its isolated B64 reduction is 0.041 ms and mixed reduction 0.293 ms; these are
local observations, not a service speedup. The original failure is retained in
[the causal record](results/moe-residual-fma-fix.md).

`DRAMA3_RESERVED_PADDING_ROUTER=1` copies real row0 router logits only into
explicitly reserved decode padding rows. Real full top-8 routing, native FP8
activation/scales and generic callers are unchanged. Production source `463d837`
passes 2574 exact checks, including fused-residual compatibility, mixed and small
batches. Fixed-state active58/B64 improves 2.52%, active63 improves 0.77%; full64
and mixed have no material gain. This remains opt-in without a separate service
claim. See [production gate](results/padding-router-production-summary.json).

The actual checkpoint's `vq_backbone_codebooks=1` allows the next backbone after
current semantic/code1 sampling, while eight remaining tower layers continue.
The native two-frame prototype passes 1792 exact checks and reduces measured
B64 two-frame latency 14.904→14.517 ms. Production `DRAMA3_FRAME_PIPELINE=1`
uses dedicated streams/workspaces and skips pairs crossing scheduled audio or
admission boundaries; production `a7118fb` passes a separate 3689-check exact gate, including slot
reuse, all captured capacities, max_frames and a clearly labeled synthetic
first-EOS case. This is about
0.194 ms per frame in an eligible pair, not a service gain. Actual ledger evidence
leaves at most47.58% of supported ordinary calls unblocked by known audio
boundaries; this is not observed pair coverage. Correlated chunk phases invalidate
independent-row estimates. See [prototype](results/frame-pipeline-results.md) and
[eligibility audit](results/pair-eligibility-ledger.json).

The first production capture hit a device assert because a callback-local frame
buffer was allocated on the main stream and consumed on the auxiliary stream
after its reference could expire. A one-variable lifetime diagnostic passes335
exact checks; retaining that buffer is the three-line source fix. A later
validator compared a Python attribute rebound by eager execution instead of the
captured hidden buffer. Its failure is retained; the final validator pins both
captured buffers and records their addresses. No numerical gate was relaxed.
Full service evaluation remains separate from these exact model-state gates.

Native codec resource limiting required recapturing graph nodes in GreenContext;
changing only the replay stream is ineffective. Inductor's device guard resets
the current CUDA context, so capture restores it before unchanged cuDNN calls.
NCU verifies all284 native B4/F8 nodes use actual32/48-SM assignments. Corrected
production `69d636e` passes a fixed dynamic80-request gate: all414 trace entries,
80 full PCM outputs and complete model/codec states are exact across normal,
capped-sync, capped-poll and capped-thread modes, covering46 captures and28 actual
input shapes. Original capture and capability-check failures remain retained.
See [dynamic gate](results/green-sm-dynamic-prerequisite-summary.json).

The complete native48-SM service pair onGPU0 observes4412.554→4470.317 frames/s
(+1.309%), makespan−2.937%, output frames−1.667%. TTFA p50/p95 changes
140.956/265.621→141.412/269.892 ms; meanRTF0.293374→0.289907. Both arms
score all1088 outputs: WER1.909068%/1.565771%, SIM63.902018/64.244919,
passing the frozen bounds. The candidate remains below the vLLM baseline and
6765 target. Production DP uses each arm's measured codec costs, so partitions
are not held fixed. This one pair does not establish a repeated or fixed-work
gain. See [complete SM-limit result](results/green-sm-service-summary.json).

Adjacent-bucket codec coalescing reduces graph calls but does not pass the original
bit-exact PCM gate. A separately declared native FP32 diagnostic permits a full
service quality check. The complete same-source `69d636e` feasibility pair onGPU7
observes **4513.299→4414.243 frames/s** (−2.195%), makespan+0.701%, frames−1.509%.
Both arms score all1088 outputs: control WER1.858829%/SIM64.022896; candidate
WER1.515532%/SIM64.251073. Both meet quality bounds, but no performance promotion
follows. See [complete coalescing result](results/coalescing-native-service-summary.json).

The complete native two-frame service pair (`a7118fb`, GPU6) observes
4576.835→4580.030 frames/s (+0.070%), while makespan grows1.186% and output
frames grow1.256%. Candidate TTFA p50/p95 is136.530/264.625ms and meanRTF0.291691.
Both unfiltered1088-output quality groups pass: WER2.118396%/2.277485% and
SIM64.359429/63.907606. This does not establish a useful service gain. Only159
of1710 ordinary launch decisions use pairs (9.30%, including64 warmup requests);
1216 are blocked by `waiting or pending`,193 by capacity and142 by audio
boundaries. The existing combined counter cannot determine how many blocked
requests were merely queued versus runnable. A CPU source audit identifies that
an unready waiting group can block a pair even though the normal admission
policy would continue decoding. See [service evidence](results/pipeline-service-summary.json)
and [admission audit](results/pipeline-admission-audit.md).

The artifact-only fused GDN recurrence/norm candidate is **not promoted**. Its
original exact check fails; preserving FMA behavior and declaring an independent
FP64 diagnostic lets one initial fixture pass. A new real fixture then fails
before wider layer coverage: per-head raw RRMS1.449e-3 and norm3.653e-4 exceed
the predeclared1e-4 diagnostic bound; one norm head also fails the independent
reference-error bound. The process stops without timing or a full-graph claim.
Whole-state averages had concealed these local differences. Original failures,
first-fixture pass and expanded failure all remain in
[the expanded result](results/gdn-fused-norm/expanded-result.json).

The subsequent admission predicate (`95fcb0e`) distinguishes queued waiting
from runnable admission while preserving pending/audio/max-frame boundaries.
Both arms capture identical two-frame graphs; only
`DRAMA3_FRAME_PIPELINE_DEFERRED_WAITING` changes. The full pair observes
4481.280→4583.673 frames/s (+2.285%), output frames−2.424% and makespan−4.604%.
TTFA p50/p95 changes136.515/267.085→146.277/274.848ms; meanRTF0.292684→0.289572.
Both1088-output quality groups pass: WER1.992799%/1.833710%,
SIM64.041274/64.120952. Pair use rises139/1902→625/1246 ordinary launches,
including warmup. All1245 old waiting rejections are measured non-runnable
waiting-only cases; the candidate selects483 pairs with deferred waiting.
This supports the coverage hypothesis, while the one-pair service gain remains
small and trades off first-audio latency. See
[complete admission result](results/pipeline-admission-service-summary.json).

`DRAMA3_CODEC_CACHED_LINEAR=1` (`e7cda1e`) keeps a private transformer copy whose
40 Linear matrices cache exactly the native BF16 autocast values. Original codec
parameters, FP32 norms/layer scales and reference encoding stay untouched; the
retained BF16 matrix cache is208MiB. A native codec profile exposes40 repeated
FP32→BF16 casts per graph,624MiB logical traffic and about141µs atB4/F8.
Compiler freezing alone retains all40 casts and fails its local performance gate.
Explicit caching removes them, passes273 exact local checks and observes
B1/B4/B8 F8 graph reductions12.9%/9.2%/8.1%. These are isolated feasibility
measurements, not service improvements.

Production source then passes all46 captured-shape PCM/full-state checks and
fixed dynamic80-request replay: all414 trace entries,80 full PCM outputs and
complete model/codec state match across normal-sync and cached-sync/poll/thread.
Private capture workspaces remain disjoint; the normal result also matches the
archived native reference. Generated dynamic input covers28 actual codec shapes.
The46-shape extension uses real reference codes, repeating a final valid code
where a window exceeds the recorded reference. Complete service quality remains
separate. See [local cache result](results/codec-cached-linear/summary.json) and
[production dynamic gate](results/codec-cache-dynamic/result.json.gz).

A native two-way expert partition passes135 exact intermediate/routing/output
checks but fails its local performance gate: despite overlap, complete execution
is17.59–24.68% slower than the original native MoE across three real layers.
Its extra metadata builds and static CTA grids are retained in the accounting.
The subsequent shared-routing rebase passes1373 exact checks but remains10.90–15.82% slower than native across the same three layers; it is also rejected for service. See [shared metadata](results/expert-metadata-rebase/summary.md). Cheap last-frame
and causal n-gram two-code proposals are rejected by a CPU-only census: the best
observed generated-prefix match rate is0.1011%, insufficient to amortize native
verification/state rollback. Neither result is evidence that the final service
target is impossible. See [MoE result](results/expert-partition/summary.md) and
[proposal audit](results/native-speculation-feasibility.md).


### Completed cache service and native fusion outcomes

The private cached-Linear full C64 pair (`e7cda1e`, GPU6) observes
**4726.771→4585.126 frames/s**, TTFA p50/p95
145.645/273.682→142.864/276.595ms and RTF0.288130→0.288258.
Both arms score all1088 outputs and meet the user quality bounds:
WER1.599263%/2.017918%, SIM64.412759/63.860895. Output frames grow2.547%
and makespan5.715%; this one pair establishes neither a repeated regression nor
a fixed-work improvement. The cache stays default off. Sampled process memory
increases105788→107058MiB (one-second polling, not a true allocator peak), while
process-to-readiness is50.032→68.039s. The best complete-quality observation is
now the4726.771 control, only1.0480× the frozen vLLM baseline. It still requires
43.13% additional throughput to reach6765.407. See
[complete cache service](results/codec-cache-service-summary.json).

A follow-up GDN implementation preserves native reduction/FMA layout with four
V32 warps. All180 real layer cases and six full-graph cases×eight steps match
codes and all111 state tensors exactly. Complete recurrence+copy+norm timing
is1.018778→1.083673ms (6.37% slower); production uses182 registers/thread.
One CPU layout-lifetime experiment increases allocation to185 and is stopped
before GPU execution. Removing launches is insufficient when fusion increases
resource pressure. Historical exact/numeric failures remain intact. See
[GDN outcome](results/gdn-fused-norm/warp-tiles-result-report.md).

The initial cuDNN-selection comparison is cache-confounded: native thread-local
plan lookup precedes the benchmark branch. A new thread per arm isolates that
cache. Fresh selection changes the first native BF16 PCM (maxabs0.00390625),
fails the original exact gate, and receives no timing or promotion. This does
not constitute a user WER/SIM failure. Both original result and correction are
retained in [the fresh selection result](results/codec-cudnn-fresh-thread/summary.md).

A fresh current-source (`e7cda1e`) nonblocking timeline completes1088 requests
with model union21.289s, codec union6.679s, overlap5.709s, codec-only0.970s
and neither0.613s. Mixed backbones account for4.946s over196 calls;
continuous-intake decode averages60.252 active rows by event time. Pair-aware
analysis holds both active count and graph steps fixed. These profiled numbers
are diagnostic, not a new acceptance group. They support prioritizing native
model execution rather than expecting exposed host/codec gaps to close the
whole target deficit. See [current timeline](results/current-pipeline-timeline-results.md)
and [delivery-phase accounting](results/cpu-service-ceiling/findings.md).

Fixed native work then isolates contention without changing generated lengths:
80 real requests,414 trace entries,99 identical codec groups and4177 exact
checks all pass. Model-only wall is1685.655ms; codec-only176.064ms normal versus
164.218ms cached. Serialized work1861.710→1850.100ms and overlapped work
1807.996→1800.720ms show that the local codec reduction becomes only0.402%
of combined overlap time in this fixture. Normal overlap stretches model event
union by120.786ms relative to model-only, but still beats serialization by2.89%.
Turning overlap off would be a regression here. The fixture excludes mixed
prefill, frame pairs, Engine callbacks and HTTP; it is a causal work diagnostic,
not service throughput. See [fixed-work report](results/cpu-service-ceiling/fixed-work/results.md).

A CPU prefix audit finds34.42% potentially duplicate prompt rows under ideal
cold reuse. Current FIFO admission grouping suggests only6.57% from prefixes
completed in earlier groups; the remaining27.86% requires same-group state
forking. Existing prefill resets state and attends only local K/V, so a simple
cache dictionary is incorrect. Extra whole-model traversals and new recurrent
rounding boundaries require a dedicated suffix path; no GPU gain is claimed.
See [prefix audit](results/reference-prefix/feasibility.md).

The first FC1 arithmetic prototype is retained as an exact failure. A CPU
source/layout audit maps all8192 candidate FP8 bytes to native order, but this
is not a corrected GPU pass:35/64 scales still differ by up to two ULP. The
failure is preserved while a separate corrected prototype is developed. It
has no timing or service promotion. See [FC1 localization](results/expert-fc1-native/layout-audit.json).

The corrected FC1 v2 passes its42 single-expert checks, while a subsequent
expanded test exposes a separate zero-input scale mismatch. Native SASS shows
that the candidate's `max(amax,1e-12)` floor is not part of the native FC1
quantizer. FC1 v3 removes that floor and handles the native zero boundary;
all438 expanded checks pass across15 real layer/routing fixtures plus native
zero and near-threshold synthetic references. Native FP8 checkpoint bytes,
scales, original top8 and K128/K32 arithmetic stay unchanged. Both previous
failures remain authoritative historical results. See
[FC1 v3](results/expert-fc1-native-v3/result.json.gz).

FC2's first failure is likewise a physical16-element store permutation;
the corrected logical BF16 store passes40 single-expert and306 expanded checks
across15 native fixtures. A genuinely new shared-memory activation device body
passes2063 checks, including partial0–8 rows, capture, guards and independently
generated native exceptional-value references. The component checks cover
routing counts0–64 and changing graph metadata, but do not establish a complete
resident chain or a speedup. Current FC1/FC2 shared usage is73592/74664B;
FC1's dynamic register roles consume53248 registers/CTA despite a smaller
static header, precluding a two-CTA register budget. See
[FC2 expansion](results/expert-fc2-expanded/report.md),
[shared activation](results/expert-activation-native/results.md) and
[resident implementation contract](results/expert-resident-implementation-plan.md).

A separate native GDN H/O fusion preserves BF16 intermediate boundaries and
passes57 synthetic,522 real-layer/fullgraph and1026 timed-trajectory checks per
variant. Three alternating timing repetitions per mixed shape show that the
V32 variant is0.13–0.51% slower. Preserving O's V64 tile avoids duplicated QK
work, but the main D58/P6 shape is still0.248% slower and the smaller shapes
are essentially flat. Both fail their local continuation condition and remain
artifact-only. Shared/TMEM allocation and loss of chunk parallelism offset a
simple intermediate-traffic argument; no hardware stall attribution is claimed.
See [V32 result](results/gdn-fused-prefill/report.md) and
[V64 result](results/gdn-fused-prefill-o64/report.md). No new service throughput
or task-quality group is claimed by these component experiments.

Source SHAs, experiment hypotheses, commands, raw per-request summaries,
complete quality records and failures are indexed by [files.json](results/files.json).
Gzip files retain the original uncompressed SHA256. Full WAVs and large tensor
captures remain under `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016`;
they are not duplicated into Git. No successful README example is added.
