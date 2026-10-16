# Current service budget: completed evidence and remaining opportunities

The four complete pipeline-admission / codec-cache service traces originally lacked internal timelines. That initial limitation is preserved in [the pre-profile report](findings-before-new-profile.md). Two subsequent measurements now add current evidence: the [full e7cda1e mixed/pipeline service timeline](../current-pipeline-service-ledger.json) and the [exact fixed 80-request replay](fixed-work/results.md). They have different scopes; the latter contains neither mixed prefill nor frame pipeline. Historical synchronous profiles are not used as current lower bounds.

## What the current traces establish

| Arm | Whole-service frames/s | Continuous-intake frames/s | Final drain seconds | Final-drain delivered frames |
|---|---:|---:|---:|---:|
| Strict pair admission |4481.28|4686.58|1.9930|4597|
| Deferred pair admission |4583.67|4740.80|1.5965|4121|
| Codec normal, deferred pairs |4726.77|4766.08|0.9814|3865|
| Codec cached, deferred pairs |4585.13|4760.03|1.8265|4760|

The cached-codec whole-service−2.997% result is not evidence that the native codec arithmetic became slower. Its continuous-intake delivery rate changes only−0.127%; the0.8451s longer drain accounts for68.8% of the1.2276s larger makespan, while output length rises2586frames. Different sampled outputs/rosters and independently profiled DP costs remain confounders. A local−8–13% codec time gain can be hidden by model overlap, altered resource contention, different DP grouping, or changed tail work. The full-service pair alone cannot distinguish those causal explanations; the later controlled replay establishes resource competition and a small combined cache benefit in its narrower fixed fixture. The result does not justify either a service gain or a repeatable codec regression claim.

Client intake stays at63.981 in-flight requests on average in all four arms. In the best4726.77fps arm, completion-to-next-dispatch gaps average0.376ms; summed missing occupancy equals only6.02ms at full64 concurrency during20.491s intake. This rules out client refill starvation as the source of a multi-second gap. Client in-flight includes queued, prefill, decode, codec and delivery states; it does NOT prove a64-row GPU decode batch. Likewise final PCM-to-HTTP-completion averages0.371ms, p950.237ms, so HTTP closing is not a large remaining budget. Audio waiting *before* the last PCM is not measured internally and cannot be eliminated by this observation.

Admission counters confirm a real fix: eligible pairs139/1902→625/1246 while1245 deliberately deferred waiting rejections disappear. The codec pair has603→599 selected pairs,467 with deferred waiting in each. These counters include64 warmup plus1088 measured requests, exclude mixed/prefill, and have no per-call roster or duration. Unsupported-capacity decisions142→301 in the codec pair are consistent with a different low-batch tail, but cannot quantify time spent there.

## Size of the missing budget

Best single observed4726.77→6765.407 requires+43.13% throughput, or reducing the same101529-frame workload from21.4796s to15.0071s:6.4725s /30.13% less elapsed time. An aggregate full64 delivery equivalent changes13.540ms→9.460ms, a4.080ms gap. This is accounting, not a measured model-step latency.

Of that target gap,6.0557s (~93.6%) lies in continuous intake and0.4101s in the final drain. Even impossibly deleting the entire0.9814s best-run drain cannot close it. Tail management and another sub-percent launch fusion cannot carry the target alone. The scalar6765 value does not establish hardware impossibility; the newly measured stage work still does not establish a hardware ceiling.

## Initial structural hypotheses and subsequent disposition

1. **Reduce codec-induced model stalls / eliminate overlap that only adds contention.** Codec GPU work and model MoE both consume bandwidth. If fixed-work overlap stretches model spans by several milliseconds more than the codec time it hides, a dependency-aware schedule or split execution windows may recover material time. A green-SM limit and coalescing already failed service; do not repeat either without a new fixed-work causal observation. Required experiment: identical native model sequence and identical codec inputs/DP groups, model-only vs serialized vs overlapped, then normal/cached codec. This is the approved next diagnostic.
2. **Reduce refill cost per useful decode row using a sequence-aware mixed executor.** Current mixed work joins a substantial prefill row block to64 decode capacity, then runs teacher tower and first-frame work. Per-graph static capacity, partial prompt occupancy and separate causal kernels can waste milliseconds on refill events, but their current count/time share is unknown. A meaningful redesign must reduce actual executed padding or share weight work without changing every real prompt's native layers/experts. Before coding, record actual active/padded decode rows, prompt lengths/capacity/span, per-mixed GPU spans and first-frame handoff. Prior span/compact/layout wins already applied; they are not new candidates.
3. **Keep more useful model work between host completion barriers.** Current complete_decode reads semantic tokens to CPU after each one/two-frame graph, and first-frame finish also reads output. Fixed scripted replay can isolate graph-only submission from host-control cadence. A device-resident completion/admission loop may help only if exposed inter-graph holes are measured at millisecond scale. Current client logs do not establish such holes; replacing Python with Rust alone is not a measured remedy. Existing bounded pipeline improves kernel overlap only~0.393ms per two-frameB64, so broadening its remaining eligibility cannot supply4ms per full64 frame equivalent.

No currently measured structural candidate has a proven 6.47 s service budget. The completed fixed replay below establishes codec interference but does not reveal a multi-second shortcut; this is not proof of impossibility.

## Minimal diagnostic and scope

Reuse frozen e7cda1e and the passed80-request/414-entry dynamic trace. Hold initial model/request/codec/RNG state, actual generated frames, slots/resets, codec input rows and original DP cost table/grouping fixed. One untimed correctness replay records the sequence; timed work replays do not perform per-step CPU output hashing. Compare model-only, codec-only normal/cached, serialized normal/cached and overlapped normal/cached. Preserve exact model outputs/metadata/final states and per-codec-call PCM/final codec state; skipped subsystem state must stay initial. Record wall, device-stage unions/overlap, actual roster/padding and CPU enqueue time. Resolve timing events after final join and compare one unobserved run to quantify observer overhead.

This80-request fixture uses packed prefill and single-step DecodeGraph; it contains neither current mixed prefill nor frame pipeline. It can identify codec/model interference and graph-submit exposure under fixed native work, but cannot establish the current full-service mixed fraction or target feasibility. If it reveals material interference, the next minimum current-service profile is one nonblocking timeline of the existing full1088 configuration with explicit benchmark phase boundaries, active/padded batch, prompt/capacity/span, codec groups/DP table, queued/pending/runnable counts, main/codec events and CPU submission/readback scopes. Keep that profile diagnostic-only and separately compare unprofiled fixed work before attributing a speedup.

Evidence: [reproducible CPU ledger script](audit.py), [all four delivery ledgers and input hashes](current-delivery-ledgers.json), [cached service result](../codec-cache-service-summary.json), [admission service result](../pipeline-admission-service-summary.json). All outputs and tails retained.

## Completed current-service timeline

Frozen e7cda1e, cached codec off, native flags/router/pipeline/deferred admission on: 1088/1088 successful requests, 105160 output frames, and a 22.873 s GPU event envelope. This is a new diagnostic workload, not the best 4726.77 fps arm. Its source, flags, inputs and timeline hashes are in [the current ledger](../current-pipeline-service-ledger.json). Events measure stage spans, including possible launch gaps; they are not kernel busy counters.

| Current stage/account | Observed time | Meaning |
|---|---:|---|
| Pure decode | 15.804 s / 1035 calls | Recorded one-, two- and four-step graph execution |
| Mixed backbone | 4.946 s / 196 calls | Includes prompt teacher tower and old-request backbone |
| First frame | 0.230 s | New-request first tower |
| Old-request mixed tower | 0.261 s | Completes old requests after mixed backbone |
| Initial prefill including first tower | 0.048 s | Two calls; prefill_many scope |
| Model union | 21.289 s | Non-overlapping model stages above |
| Codec union | 6.679 s | PCM conversion/copy included |
| Model/codec overlap | 5.709 s | Must not be added twice |
| Codec-only span | 0.970 s | Direct exposed codec time; not its whole possible contention cost |
| Neither model nor codec span | 0.613 s | Includes submit/other work; not all idle |

During continuous intake, pure decode's event-time-weighted active count is **60.252**, versus client in-flight occupancy near 64. Whole-run step-weighted active count falls to **51.113** because the complete drain is retained. Pure decode executes **94559 actual request rows**, plus **10988 old-request mixed rows**; recorded pure padding is **5.82% over actual rows**, mixed padding **12.85%**. Keeping every batch at 64 is not an available free optimization: queued/prefill/codec states and completed requests account for occupancy, and changing the last tail alone cannot solve the intake gap.

Mixed work admits **1079 prompts / 183228 actual tokens / 209664 prefill-capacity rows**, or **14.43% padding over real tokens** (12.61% of allocated capacity). Mean prompt count is 5.505 and mean actual token count 934.84; 195/196 spans are already 256. D64/P1024 is 65 calls at 24.207 ms; D64/P1280 is 59 at 27.200 ms; D64/P1536 is 18 at 30.173 ms. Another span reduction is not a new broad opportunity. Eliminating token padding would affect only part of each mixed graph, so 12.61% of the whole 4.946 s is already an optimistic linear estimate, not a measured saving; the actual kernel traffic/compute relation is nonlinear.

Codec has 2206 calls across 972 nonempty emissions. Same-frame grouping alone requires 2064 calls; only 142 additional calls come from partitioning, while 1092 extra groups are due to frame-length diversity. Therefore reversing the already-tested DP/coalescing decisions cannot remove most calls. A genuinely ragged native codec would need to combine different frame buckets while preserving per-request causal state and PCM exactly. Its service budget remains exposed codec plus any removed contention, not all 6.679 s.

For equal active B64 and step count, observed event means are **9.299 vs 8.562 ms** (one step, with/without codec overlap), and **18.596 vs 16.762 ms** (two steps). Matching intake phase, active count and steps across 57–64 yields **1.394 s associated excess**, but context/cache/resource history remain uncontrolled; this is not causal or a guaranteed recoverable budget. The fixed replay below supplies causal control in a different fixture.

CPU decode submission totals only **72.875 ms**, while first-frame CPU scope totals 5.106 s largely because it waits on the preceding mixed graph. It must not be counted again as Python overhead. The total 0.613 s outside model/codec spans is also not a multi-second interpreter budget. Replacing the engine language alone has no supporting large budget in these measurements.

## Completed fixed-work control

[Raw and summarized fixed replay](fixed-work/results.md) passes **4177 exact checks**, preserves archived 414 trace entries and all 80 request PCM streams, and fixes generated codes, model state, codec groups, row resets and DP costs. Codec-only normal/cached takes **176.064/164.218 ms (-6.73%)**; overlapped model+codec takes **1807.996/1800.720 ms (-0.402%)**. Codec competition stretches model union **1684.810→1805.596 ms (+7.17%)**, but full serialization takes **1861.710 ms**, so overlap remains beneficial. The event/no-event comparison differs 0.124% in this single sequence. No mismatch occurred; source remains clean and GPU6 is released.

This resolves the apparent codec paradox without claiming the earlier complete-service -3% is deterministic. A real isolated improvement mostly changes overlapping work, and its smaller delivery effect is readily hidden by variable sampled lengths, roster/DP scheduling and tail work. The fixed fixture cannot quantify the current service's exact codec counterfactual.

## Priority after these measurements

1. **Core native decode dataflow and expert weight residency remain the route with sufficient scale.** Current pure decode alone is 15.804 s; required prompts and first-frame work add 5.485 s before codec-only and other spans. There is no identified host or codec-only scheduling trick that removes the 6.47 s best-run gap. A native expert-resident kernel or another actual reduction in the repeated core model passes needs independent exact arithmetic/state validation and measured kernel savings; the parallel expert-resident investigation is therefore more material than another small scheduler sweep. Existing ~one logical FP8 weight read and bandwidth evidence still forbid assuming easy gains.
2. **Mixed prompt computation is the next measured structural budget, but needs a real native algorithm improvement.** Halving the entire mixed+first+tower 5.438 s would save at most 2.719 s before secondary effects, still far short of the current diagnostic's 7.353 s target gap. Packing away the remaining 12.61% allocated prompt padding is much smaller. Do not repeat compact chunks, span256, layout fusion or expert-layout experiments already tested. A new proposal should name which large projection/expert/recurrence work is removed or shared and validate every real token/state.
3. **Codec-aware placement is a bounded supporting direction, not the main target route.** Full serialization already loses in fixed work. A finer phase schedule would need an unchanged current mixed/pipeline workload proving that it reduces contention more than it exposes codec latency, without changing chunk ramp or TTFA contracts. The current 0.970 s exposed codec and 1.394 s uncontrolled overlap association do not add into a guaranteed saving. The fixed fixture's ideal zero-codec counterfactual removes only 122.341 ms / 6.77% of its wall.

No proven candidate yet supplies the missing multi-millisecond equivalent per full64 step. The isolated B64 rate (~7734 fps) exceeds 6765 but omits mandatory prompts, context variation, partial batches and audio; it neither proves the service target achievable nor proves it impossible. Current evidence directs further native kernel/dataflow work rather than claiming a theoretical ceiling or changing the user's quality/computation contract.

Independent timeline review: [15 accounting checks pass](current-timeline-review.md). The legacy 815 field is extra graph steps, not request rows; initial prefill includes its first tower. Original parent ledger/raw remain unchanged.
