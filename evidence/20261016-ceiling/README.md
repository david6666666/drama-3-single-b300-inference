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
Correction: the FC1 source-role hints sum to53,248, but its compiled
CUBIN has REG32 and no SASS register-reconfiguration instruction. The former
claim that this hint sum precludes two CTAs is withdrawn. See
[the final-machine-code audit](results/register-lowering-audit/report.md). See
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


## Resident-chain composition and M128 resource experiment

The complete FC1→native activation→FC2 resident prototype passes489 exact
checks for one selected layer0/expert205 at valid row counts0–8: five native
boundaries, eager/graph, guards, input/weight/metadata/RNG ownership and an
explicitly oracle-assisted native finalizer. Its actual compiled launch uses
78,464dynamic+1,024static bytes perCTA, zero spills, and one outer-owned TMEM
allocation through all stages. This is **one-expert correctness only**;
full256 routing, nondebug timing and service acceptance remain separate.
The initial attempt failed at the host ABI before executing its candidate kernel;
[the failure](results/expert-resident-chain/chain-launch-result.json) and
[the corrected single-expert result](results/expert-resident-chain-v2/summary.md)
are both retained. The independent CuTe activation helper also passes2066 checks;
its earlier C++ helper result was not reused as proof for the new implementation.

A separate M128 experiment targets the resource limitation of the native M64
two-epilogue organization. The first FC1 gate fails: four TMEM slices were stored
as adjacent channels. A CPU-only mapping diagnostic identifies the fixed
32-channel permutation, without passing the original checker or changing its
expected values. FC2 uses the corresponding corrected four-slice BF16-to-shared
store, preserving K32 reductions, K128 scale boundaries, original FP8 weight
bytes/scales and native BF16 output conversion. It passes39 single-expert checks
and306 expanded checks across all256 experts at layers0/19/39 and15 original
routing distributions, including graph reuse with changing metadata.
See [FC2 expanded evidence](results/expert-m128-fc2-expanded/report.md).

M128 uses384 threads versus512 for the M64 prototype. Its FC1/FC2 CUBINs
report REG32/REG31, shared totals74,616/74,664 bytes and zero stack/local spills.
The source48/160 register hints do not survive as register-reconfiguration SASS
in either prototype. The vendor native FC1 differs: REG128 and eight
`USETMAXREG` instructions. Thus53,248 versus32,768 source-hint sums do not
establish an occupancy transition. That earlier interpretation is withdrawn;
[raw disassembly and hashes](results/register-lowering-audit/result.json)
separate source hints, actual code and unmeasured occupancy.

The GDN H/O negative results were independently recomputed from raw evidence.
The reports now correctly describe pre/post GPU process checks; no continuous
process audit establishes full-interval exclusivity for those two runs. This
correction does not turn either failed continuation gate into a performance win.
See [independent review](results/gdn-fused-prefill/independent-review.md).

## Complete resident and M128 calls: local performance gates fail

The full256-expert resident chain subsequently passes1,371 native-boundary
checks across15 fixtures. A separate build removes global debug exports and
passes1,262 correctness/ownership checks, including an empty launch. Three
complete-call repetitions per fixture then fail the declared performance gate:

| Real fixture | Vendor µs | Resident µs | Resident latency increase |
| --- | ---: | ---: | ---: |
| Layer0 |121.299|138.474|14.16%|
| Layer19 |100.507|122.128|21.51%|
| Layer39 |81.407|111.619|37.11%|

All12 pressure cases also regress. Calls include live original native routing
metadata, both projections, activation and ordered finalizer. Common input
quantization and logits-to-top8 routing are excluded. The vendor retains PDL;
the candidate currently disables it. This is an implementation comparison, not
an isolated measurement of fusion, MMA shape, or synchronization. The remaining
shared-buffer canary initialization is included in candidate time. Removing
intermediate global traffic did not outweigh the complete implementation cost;
no measured hardware-stall attribution or service gain follows.
See [full chain](results/expert-resident-full/summary.md) and
[complete-call timing](results/expert-resident-timing-v2/summary.md).

M128's corrected FC1 passes542 selected-expert and447 expanded checks. Its
combined global-buffer body passes1,253 checks and another626 during timing.
The three real cases reduce vendor latency by1.304%,0.569%,2.871%; each needed
at least5%. The12 pressure cases do not regress, but neither an average nor the
largest synthetic gain substitutes for the frozen real-case gate. The full
call therefore also stops before runtime/service integration. See
[M128 complete-call report](results/expert-m128-body-timing/report.md).

Both harnesses preserve initial failures caused by using archived physical
metadata for newly populated native metadata. Native atomic routing may legally
reorder rows within an expert. Corrected checks first prove the live map's
bijection, original-token/expert ownership and counts, then compare logical
values byte-for-byte. They do not infer permutations from output values or
relax precision tolerances. Frozen kernels are unchanged by these checker fixes.

These tests preserve original FP8 bytes/scales, all top8 experts, K128 scale
boundaries and native BF16/FP32 arithmetic. They add no complete1088-request
quality group or C64 throughput result. The best complete-quality observation
remains4726.771 frames/s;6765.407 and the all-concurrency quality gates remain
unmet. Register source-hint occupancy claims are corrected explicitly above.


## M128 PDL enabled: exact pass, feasibility gate fails

Enabling PDL for the candidate GEMMs, activation and finalizer passes1498 exact
checks, including all15 fixtures, changing live metadata, empty work and recovery
with the same graphs. Captured graphs have actual programmatic dependency edges;
a kernel-node attribute query returns `CUDA_ERROR_INVALID_VALUE` and is recorded,
not treated as successful attribute evidence. Native baseline PDL remains enabled.

One preliminary AB measurement per real layer gives120.781→118.380µs,
100.754→96.892µs and81.966→76.790µs:1.988%,3.832%,6.315% latency reductions.
Not all real cases reach5%, so the run stops before three-repetition timing or
runtime integration. The12 pressure cases pass arithmetic checks but their
performance is **NOT RUN**; a vacuous raw stress flag is not a measured pass.
These single-round observations must not be causally compared with the earlier
separate PDL-off experiment. See [PDL report](results/expert-m128-pdl/summary.md).

CPU inspection identifies a narrower follow-up: the native FC2 weight producer
can precede a consumer dependency wait, while this prototype waits at entry.
An explicit metadata-completion dependency and retained consumer waits would be
required to exploit independent weight prefetch safely. That motivated the separately validated specialization below. See [synchronization contract](results/pdl-memory-contract.md).


## Metadata-ready FC2: safe earlier loads, insufficient improvement

The restricted FC2 removes only its global metadata wait. Every actual metadata
producer receives a DEFAULT full-memory graph edge to FC2; the original activation
PROGRAMMATIC edge and B/SF consumer waits remain. CPU checks cover26 configuration
and44 graph cases. Final PTX/SASS places weight TMA before activation-consuming
waits. These facts establish the intended dependency separation, not a speedup.

The full GPU gate passes1516 byte/ownership checks across15 fixtures, including
same-graph0→positive→0 metadata restoration. One complete-call screening gives:

| Real layer | Native µs | Candidate µs | Reduction |
|---|---:|---:|---:|
|0|120.748|117.452|2.729%|
|19|99.425|96.260|3.183%|
|39|79.944|76.630|4.145%|

All miss the predeclared5% gate. No three-repeat timing, pressure-case timing or
runtime promotion follows; pressure performance is explicitly NOT_RUN/null.
The130 frozen files still match after execution. Trusted/private graph ownership
is part of the contract; topology snapshots do not inspect kernel argument bytes.
See [the complete report](results/expert-m128-pdl-metadata-ready/summary.md) and
[conditional outer-graph integration design](results/metadata-ready-runtime-integration-design.md).

## Bounded native-dtype Lt investigation: no selected replacement

The first strict BF16 capability filter selects no candidate. Its apparent whole
sequence timing difference is A/A variation. Both tested descriptors request
BF16 A/B/D and FP32 compute, but cuBLAS13.1.1 reports INPUT_16F. Independent C++
and actual descriptor getters reproduce the discrepancy. Eight full-output range
probes match rounded FP64 exactly, including values beyond FP16 range. This does
not resolve the documented capability semantics or permit a generic filter waiver.

A separate investigation binds the exact two algorithm blobs, shapes, library
and descriptor settings. All84 real calls pass full eager byte equality and84
captured-graph output/input/weight/guard/RNG checks. The30-call non-split sequence
is0.331940→0.331992ms; the54-call FP32 split-K sequence0.412405→0.412599ms.
Neither is selected. The later223-call sequence uses F.linear on both arms, so
its0.0265ms difference is also A/A variation. No0.3ms budget, repeated gain,
native runtime qualification or service improvement is established.

The223 calls are intercepted `drama3.model.linear` operations, excluding direct
matrix operations elsewhere. An initial C++ probe accidentally resolved system
cuBLAS13.7 through LD_LIBRARY_PATH; the corrected child-only library path pins
13.1.1, matching Python's actual mappings throughout. Both probes remain recorded.
Package metadata or RUNPATH alone does not prove the loaded library identity.
See [dtype provenance](results/native-lt-dtype-contract/summary.md) and
[bounded real-input result](results/native-lt-real-contract/summary.md).

The best complete-quality C64 observation remains4726.771frames/s. These local
negative results do not prove6765.407 impossible, and they do not satisfy the
all-concurrency1.5× performance and WER/SIM acceptance requirements.

## Current hardware counters and a bounded route-address experiment

Exactly eight NCU node records on GPU2 cover L0/real and L39/real, FC1/FC2,
native versus metadata-ready M128. All404 correctness/ownership checks pass;
each selected kernel uses four replay passes, clock/cache control remain off,
and original predecessors/PDL remain in the complete graph. Node replay still
serializes launches: these timings are not complete-call, overlap or service
measurements. Binary NCU reports remain in the local artifact directory;
the exported metric CSV, commands, source, hashes and summaries are archived here.
The first launcher stopped after two records because NCU2026.3 emitted `.ncu-repz`
instead of `.ncu-rep`; v2 imports those two and samples only the remaining six.
No data were silently replaced or resampled.

| FC1 observation | L0 native → candidate | L39 native → candidate |
|---|---:|---:|
| Active CTAs per active SM cycle |1.00 → 1.95|1.00 → 1.94|
| Eligible warps per active cycle |0.30 → 0.20|0.36 → 0.26|
| Issue-active percent |22.71 → 16.41|26.83 → 20.08|
| DRAM reads MB |432.35 → 432.25|241.24 → 246.83|

Higher residency is real, but it does not yield proportionally more issuable
work. Across all four paired nodes, DRAM read changes range from−0.154% to+2.318%;
there is no large repeated-weight traffic saving. Long-scoreboard ratios point
toward investigating ordinary load dependencies, but their issue-active
normalization and aggregation cannot identify a particular load or establish
TMA starvation. See [all17 metrics and limits](results/expert-m128-ncu-node-v2/summary.md).

A specific source audit found that the FC1 scale producer reloads the same
route-map row for each of16 K128 blocks. The artifact caches that row once per
work tile while preserving the live metadata, masks, PDL and all FP8/FP32/BF16
boundaries. AOT confirms the repeated load disappears, FC2 is unchanged, register
and shared-memory allocation stay the same and there are no spills. All1516
exact checks pass. One AB complete-call screen gives120.871→117.194µs,
99.385→96.603µs and80.055→76.704µs, improvements3.04%/2.80%/4.19% versus the
original vendor baseline. All miss the5% continuation gate; no repetitions,
stress timing or runtime integration follows. Separate runs against the old
candidate cannot be subtracted to claim an isolated route-cache gain.
See [route-cache evidence](results/expert-m128-route-cache/summary.md) and
[the memory/dependency audit](results/native-small-expert-memory-audit.md).

## Original mixed causal-prefix identities and compact native parts pass

The native e7cda1e mixed path preserves byte-identical normalized MoE input rows
for three reconstructed real prompt cases: one group, its reversed order/slots
through the same graph, and a changed-reference group. All74120 row-pair/layer
comparisons across40 layers are exact;3360 nonshared control comparisons differ.
The matcher requires complete causal11-column prefix identity, exact position,
conditioning and original zero initial state. Decode and original padding rows
are not merged. The group membership is a FIFO proxy whose token totals match
the recorded timeline, not a recovered actual admission list.

A separate compact→native parts→expand gate now passes5393 checks. It retains
every decode and original padding row, compacts1600 rows to fixed capacity1088,
calls the unchanged native `model.moe_parts`, and expands all three outputs.
All40 layers in three real cases pass eager and same-graph byte comparisons of
FP8 bytes/scales/logits, canonical top8, routed/shared outputs and shared gate.
All111 model/request states, read weights, guards and RNG remain unchanged.
The120 reconstructed full-input hashes match the earlier observer. The first
harness's scalar-byte-view failure is preserved; v2 changes only that comparator
and exercises scalar/signed-zero/empty CPU tests before rerunning the whole gate.

This component gate is followed by a separate complete mixed-graph screen below.
Read the [CPU row budget](results/native-row-reuse-audit/proposal.md),
[actual alias checks](results/native-row-reuse-audit/alias-results.md) and
[compact native-parts checks](results/native-row-reuse-exact-v2/results.md).
Theoretical removed prompt rows are not a service speedup, especially while
participating expert weights still require their first reads. Raw hidden tensors
remain local; published manifests retain their hashes and original paths.

## Ordered finalizer plus residual normalization: exact, below its budget

A separate GPU4 operator artifact fuses native top8 finalization into the existing
shared-expert combination/residual RMSNorm while preserving every rounding
boundary. All1355 provenance, tensor, graph, guard and RNG checks pass over15
original FC2 fixtures and changing physical/missing-row maps. One AB100-replay
screen saves1.057/1.036/1.801µs for the three real layers, below the predeclared
2.5µs per-call gate. This is not complete-MoE or model timing, and no integration
or serving test follows. Both graphs include an identical residual restore.

Original setup failures are retained, including an occupied-GPU preflight, UUID
spelling check and a wrong native-reference ABI boolean. The archived native
output check caught that unweighted control before candidate comparison; after
correcting `use_routing_scales_on_input=False`, the unchanged strict checks pass.
See [finalizer result and all failure provenance](results/native-finalize-norm-v3/summary.md).
The best complete-quality C64 observation remains4726.771frames/s;6765.407 and
all seven concurrency/quality gates are still unmet.

## Instruction sampling remains unresolved; tower counts are not a savings budget

The separately frozen PC-sampling run stops after one native FC1 record with
INCONCLUSIVE status. It passes80 exact checks and records4815 samples,3503 long
scoreboard samples, with no reported overflow or dropped bytes. The report uses
loaded program addresses while the frozen disassembly uses function-relative
offsets. All445 sampled addresses fail the direct-offset mapping gate;21 return
no SASS instruction. The two pass fields also disagree. No base is guessed from
instruction text or minimum sampled address, and the candidate is NOT_RUN.
A later CPU audit finds an explicit report-owned function base, but the21
missing PCs remain outside the function and the pass discrepancy is unresolved.
No attribution to a specific load or native/candidate comparison is established.
See the [preserved report](results/expert-m128-fc1-pcsampling/summary.md) and
[read-only mapping follow-up](results/expert-m128-fc1-pcsampling/pc-base-independent-audit.md).

A CPU-only audit of the older three-replay B64 trace identifies a1.332ms tower
span,373 kernels and only0.014ms of uncovered gaps. Summed kernel time overlaps
and must not be subtracted from wall time. The whole13-layer FFN sequence costs
0.388ms, including both GEMMs; saving0.3ms would require removing77% of that
whole segment while retaining its arithmetic. The223 intercepted model-linear
calls omit12 direct tower matrix operations. This audit finds no justified
new small-fusion budget above0.3ms and makes no current-service roofline claim.
See the [scope and complete accounting](results/native-tower-budget-audit/summary.md).

## Full mixed causal-prefix compaction: exact, below continuation threshold

The unchanged native MixedGraph is compared with an artifact that gathers1088
rows, runs the original full-top8 MoE and expands its three parts to1600 rows
at each of40 layers. Attention, GDN updates, teacher-forced tower, graph output
and all111 model/request state tensors remain byte exact in957 checks across
three real proxy cases, including reversed order and changed conditioning.
Each warmup and timed replay restores the same complete initial state before
its start event. The graph includes gather/expand/backbone/teacher; independent
postgraph sampling towers, codec and host preparation are outside this boundary.

| Real proxy case | Native median ms | Candidate median ms | Reduction |
|---|---:|---:|---:|
|190|29.130736|27.997727|3.889%|
|190, reversed order and reused slots|29.136928|27.994192|3.922%|
|192, changed conditioning|29.300735|28.023808|4.358%|

This is one AB100 screen per case, not three independent repetitions. All three
miss the frozen5% gate. No runtime integration or service comparison follows.
The candidate retains704027456 extra buffer bytes. Removing redundant adapter
copies is a separate hypothesis; it does not retroactively qualify this result.
See [exact results, state restoration and timing scope](results/native-row-reuse-mixed/results.md).

The separate [full-prefix DAG CPU audit](results/prefix-dag-cpu-audit/summary.md)
explains why exact MoE-row sharing does not automatically share recurrent state:
a common25-row prefix ends inside the native64-row GDN chunk, and the teacher's
next-row labels shorten its safe common prefix. A full per-layer fork remains a
new implementation. Its optimistic row-linear estimates are not measured gains;
the entire current mixed-plus-prefill interval is smaller than the service gap.

## Fused prefix adapter passes complete mixed graph continuation

A new adapter keeps the native compact MoE parts, but lets the existing
combination/residual/RMSNorm kernel read them through the inverse row map. The
residual and normalized output retain all1600 original rows; its arithmetic and
rounding are unchanged. One slot gather per replay replaces40 repeated gathers,
and direct inverse reads remove120 component-expansion launches. This reduces
adapter launches from200 to41 and retained candidate buffers from704027456 to
441087536 bytes. Logical eliminated copy bytes are not measured DRAM savings.

After603 exact checks, its single AB100 screen passes the5% gate on all three
real cases. A separate frozen AB/BA/AB repetition protocol then passes2430 checks,
including complete outputs,111 states, RNG and guards after every timing pair.
Every replay starts from the same restored initial state in both arms.

| Real proxy case | Three-round median latency reduction |
|---|---:|
|190|6.2555%|
|190, reversed order and reused slots|6.2599%|
|192, changed conditioning|6.7568%|

All nine pairs exceed5%, ranging6.2256–6.7673%. These are same-process, same-capture
complete mixed-graph repetitions, including the teacher tower, but excluding
postgraph sampling towers, codec, host admission preparation and state restores.
They do not establish service throughput or arbitrary-shape correctness. Actual
JIT binaries are recorded separately from AOT; both deployed specializations
have32 registers and zero spills, while their actual CUBINs differ from offline
binaries. Preparation costs are recorded separately. Runtime integration and
expanded real scheduling validation are subsequent work.

See the [single screen](results/native-row-reuse-fused-adapter/results.md) and
[three-round full results](results/native-row-reuse-fused-rounds/results.md).
The original expanded-adapter continuation failure remains unchanged.

## Transparent lossless allocation is exact but misses its performance gate

A context-free driver query confirms generic memory compression support on the
designated B300. A bounded CPU survey reads27 original FP8 matrices and preserves
all raw hashes; CPU entropy and zlib results motivate investigation but do not
predict hardware compression or performance. A subsequent GPU4 allocation gate
passes47 assertions: actual allocation flags0/1, all original FP8 bytes,
captured reads with changing masks, Tensor-view lifetimes and cleanup. Five
allocations are released with zero cleanup errors and no remaining process.
This first gate establishes allocation correctness only. A separately frozen
complete native MoE screen subsequently passes3417 checks over15 fixtures,
including live routing, all intermediate boundaries, top8 finalization, guards,
read-only weights and RNG. All12 allocations report their requested actual
compression flag and are destroyed with zero cleanup errors.

The candidate changes only backing allocation; it must preserve original FP8
bytes, scale tensors, layouts and native kernels. See the
[CPU scope](results/native-lossless-weight-audit/summary.md) and
[allocation/lifetime gate](results/native-compressed-vmm/micro-summary.md).


| Fixed native MoE scope | VMM NONE µs | VMM GENERIC µs | Latency reduction |
|---|---:|---:|---:|
|L0|120.874557|120.099201|0.6415%|
|L19|100.473280|100.696316|−0.2220%|
|L39|80.344324|80.093756|0.3119%|
|Three-call sequence|295.408325|295.115204|0.0992%|

These are one AB100 observation, not a repeated speedup claim. The sequence
misses its predeclared5% threshold; no further timing rounds or integration
follow. Both arms execute the original native body with tactic(8,5), PDL and
unchanged FP8 weights/scales. Timing includes native metadata, FC1, activation,
FC2 and finalization; common quantization/logits routing are excluded equally.
The sequence uses three fixed layer inputs, not a three-layer backbone dataflow.
Allocation capacity and CPU entropy are not hardware compression ratios. No
traffic counters identify why the allocation change fails to help. See the
[complete native result](results/native-compressed-moe/summary.md).

A separate [CPU zero-structure audit](results/native-zero-structure/summary.md)
finds most complete zero channels in its L0 samples, with very few in L1 and none
in L2/L19/L39. It preserves both positive and negative zero bytes and all45
roundtrips. Fixed samples do not establish a model-wide compression ratio or
support channel pruning. Neither this diagnostic nor local graph improvements
change the best complete-quality C64 observation of4726.771frames/s.


## Full FP32 scale-tile cache: exact, below continuation threshold

A separate FC1 producer stages all original activation and weight scale groups
once per work tile in576 bytes of shared storage. It uses AsyncThread publication
for ordinary shared stores, preserving the original activation gather, weight
TMA, MMA/FP8 boundaries and metadata-ready FC2. CPU address/bit-copy checks pass
1,113,949 assertions, and the deployed FC1/FC2 binaries match their AOT binaries.
FC1 uses75,192 shared bytes, REG32 and no spills. These resource facts do not
establish occupancy or that scale loads dominate the runtime.

The GPU gate passes1518 checks over15 fixtures, including live routing, original
intermediate boundaries, top8 finalization, guards, RNG and empty→positive→empty
replay for the GEMM-only empty probe. A further148 checks pass after timing.
One complete native-body AB100 screen observes latency reductions3.4505%,4.1728%,
and4.4240% onL0/L19/L39. Every case misses the frozen5% continuation gate; no
additional rounds, stress timing or service integration follows.

The comparison retains original native tactic(8,5) and PDL. It times metadata,
FC1, activation, FC2, finalization and required dependency edges; common input
quantization/logits routing are excluded equally. The scale producer is the
isolated source change relative to its prior candidate; comparison to vendor
native is a complete implementation comparison. Separate historical screens
cannot be subtracted to attribute the difference to scale caching. See the
[raw result and loaded binary identities](results/expert-m128-sf-tile-cache/GPU-summary.md).


## Finite production prefix reuse: complete graph correctness passes

Source `345c6a1` adds default-off `DRAMA3_MIXED_ROW_REUSE=1`, requiring the
existing mixed backbone, span256 and fused native MoE residual. Five graphs are
built before readiness: D64/P768/K576, P1024/K832, P1280/K1088 and
P1536/K1088 orK1344. Unsupported shapes or insufficient equal-prefix rows use the
original graph. Full eleven-column causal prefixes are compared on CPU; immutable
prompt bytes reject stale same-length conditioning before any graph input write.
All decode/padding rows and original top8 arithmetic remain.

The first production gate passes eight real cases before detecting unequal GDN
state only in private padding slot143. Its two arms had different padding input
histories: a reused original graph retained55 prior prompt rows,53 byte-different,
while the compact graph still held initial EOS/zero rows. The original failure
and complete tensors are retained. Production code is unchanged by the fix: a
separately frozen v2 harness initializes identical full inputs in both arms and
checks their complete bytes before and after replay, retaining every state check.

The corrected gate passes **9974/9974 checks in30 cases**, including original
A/A determinism, all five compact buckets, ascending/reverse capture order,
interleaved original/compact graphs sharing a pool, complete111 model/request
states,106 nonparticipating state tensors and prior returned-output clones.
Changed slots, partial active batches, prompt boundaries, cancellation before
admission, stale conditioning and all original fallback paths are included.
Ordinary decode/frame-pipeline graphs retain their real separate pools.

Production temporary tensors use the existing graph pool rather than retaining
forty artifact-private copies. Five attention workspaces alone add640MiB; the
correctness harness's125.13GB allocated peak includes two graph sets and state
snapshots and is not a serving-memory measurement. CPU plan construction plus
full-content binding costs median0.104ms/p950.178ms in171 supported-shape proxy
groups; H2D and other preparation are outside this CPU probe. Actual service
walltime must include all of them. This gate qualifies correctness only; C64
throughput, full quality and all-concurrency acceptance remain separate.


See [production design](results/native-row-reuse-runtime-design/implementation-proposal.md),
[preserved first failure](results/native-row-reuse-runtime-exact/summary.md) and
[complete corrected gate](results/native-row-reuse-runtime-exact-v2/summary.md).


## Final service result and user-requested pause (2026-10-16)

Optimization is paused. The complete production prefix gate passes 9974 checks
across 30 cases, but the subsequent full service feasibility pair fails its
throughput gate. All 1088 outputs in both arms are scored; none are filtered.

| Row reuse | Frames/s | TTFA p50 / p95 ms | Mean RTF | WER % | SIM |
|---|---:|---:|---:|---:|---:|
| Off | 4674.814977 | 148.562946 / 274.704463 | 0.290890 | 1.465293 | 64.188428 |
| On | 4534.131006 | 151.265127 / 273.353699 | 0.290618 | 1.682994 | 64.202021 |

Source `345c6a18f200853489a9ac8a86bbe0682ebe484d`, same GPU6/C64/1088 requests,
one AB pair. Candidate/control throughput 0.969905981 misses the frozen 1.01
continuation gate. Both quality groups pass the unchanged same-C vLLM bounds.
No three-round service trial follows, and row reuse remains default-off.
The single best complete-quality observation stays 4726.771495 frames/s,
below the 6765.407263 target. All-concurrency acceptance remains INCONCLUSIVE.

The candidate activates in 130/200 lifetime opportunities, removes 36096 rows,
and records 29.439 ms CPU plan construction. Output lengths differ and the
last-dispatch-to-completion tail rises from 1.046 to 1.844 s. This client accounting
does not establish a GPU cause; the full tail remains included. A local graph
latency reduction is therefore not promoted to a service speedup.

All 1808 frozen dependency hashes were rechecked. Four service/quality runners
exit successfully and record empty selected-GPU state after cleanup. Sampled
ownership cannot rule out subsecond interference. See
[summary](results/native-row-reuse-service/summary.md),
[complete results](results/native-row-reuse-service/summary.json),
[protocol](results/native-row-reuse-service/protocol.json) and
[archive status](results/native-row-reuse-service/SAFE_TO_ARCHIVE.json).

The fixed eight-lane SIMT mapping is closed at CPU/AOT analysis without a GPU
run: the conditional instruction-issue budget is already about 5.44 ms for the
conversion/FMA subset, before other work, versus an approximately 4 ms historical
native GEMM budget. This is a mapping-specific estimate, not a universal physical
limit. Only 384 selected FC2 BF16 outputs were checked against native values;
FC1's original FP8 epilogue/full MoE equality is not established. See the
[CPU report](results/native-fp8-simt-cpu-audit/report.md) and
[independent review](results/native-fp8-simt-independent-review.md).

README architecture and the available three-framework tables now document
current results at the user's request. The historical SGLang workload is labeled
separately; neither a matched three-way victory nor the target is claimed.
