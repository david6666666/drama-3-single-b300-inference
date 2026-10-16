# Current service budget: observations and missing measurements

CPU-only inputs are the four complete pipeline-admission / codec-cache request traces, their actual selector shutdown counters, launch metadata and full quality summaries. No current service arm enabled TimelineProfiler. Consequently, these runs do not measure internal effective decode batch, mixed-prefill time share, codec GPU overlap, or CPU graph submission latency. Historical rich-timeline sourceab6da53 is a separate configuration and cannot supply a current lower bound.

## What the current traces establish

| Arm | Whole-service frames/s | Continuous-intake frames/s | Final drain seconds | Final-drain delivered frames |
|---|---:|---:|---:|---:|
| Strict pair admission |4481.28|4686.58|1.9930|4597|
| Deferred pair admission |4583.67|4740.80|1.5965|4121|
| Codec normal, deferred pairs |4726.77|4766.08|0.9814|3865|
| Codec cached, deferred pairs |4585.13|4760.03|1.8265|4760|

The cached-codec whole-service−2.997% result is not evidence that the native codec arithmetic became slower. Its continuous-intake delivery rate changes only−0.127%; the0.8451s longer drain accounts for68.8% of the1.2276s larger makespan, while output length rises2586frames. Different sampled outputs/rosters and independently profiled DP costs remain confounders. A local−8–13% codec time gain can be hidden by model overlap, altered resource contention, different DP grouping, or changed tail work. Current data cannot distinguish those causal explanations. The result does not justify either a service gain or a repeatable codec regression claim.

Client intake stays at63.981 in-flight requests on average in all four arms. In the best4726.77fps arm, completion-to-next-dispatch gaps average0.376ms; summed missing occupancy equals only6.02ms at full64 concurrency during20.491s intake. This rules out client refill starvation as the source of a multi-second gap. Client in-flight includes queued, prefill, decode, codec and delivery states; it does NOT prove a64-row GPU decode batch. Likewise final PCM-to-HTTP-completion averages0.371ms, p950.237ms, so HTTP closing is not a large remaining budget. Audio waiting *before* the last PCM is not measured internally and cannot be eliminated by this observation.

Admission counters confirm a real fix: eligible pairs139/1902→625/1246 while1245 deliberately deferred waiting rejections disappear. The codec pair has603→599 selected pairs,467 with deferred waiting in each. These counters include64 warmup plus1088 measured requests, exclude mixed/prefill, and have no per-call roster or duration. Unsupported-capacity decisions142→301 in the codec pair are consistent with a different low-batch tail, but cannot quantify time spent there.

## Size of the missing budget

Best single observed4726.77→6765.407 requires+43.13% throughput, or reducing the same101529-frame workload from21.4796s to15.0071s:6.4725s /30.13% less elapsed time. An aggregate full64 delivery equivalent changes13.540ms→9.460ms, a4.080ms gap. This is accounting, not a measured model-step latency.

Of that target gap,6.0557s (~93.6%) lies in continuous intake and0.4101s in the final drain. Even impossibly deleting the entire0.9814s best-run drain cannot close it. Tail management and another sub-percent launch fusion cannot carry the target alone. The scalar6765 value does not establish hardware impossibility; current stage work and overlap remain unmeasured.

## Structural hypotheses worth measuring before implementation

1. **Reduce codec-induced model stalls / eliminate overlap that only adds contention.** Codec GPU work and model MoE both consume bandwidth. If fixed-work overlap stretches model spans by several milliseconds more than the codec time it hides, a dependency-aware schedule or split execution windows may recover material time. A green-SM limit and coalescing already failed service; do not repeat either without a new fixed-work causal observation. Required experiment: identical native model sequence and identical codec inputs/DP groups, model-only vs serialized vs overlapped, then normal/cached codec. This is the approved next diagnostic.
2. **Reduce refill cost per useful decode row using a sequence-aware mixed executor.** Current mixed work joins a substantial prefill row block to64 decode capacity, then runs teacher tower and first-frame work. Per-graph static capacity, partial prompt occupancy and separate causal kernels can waste milliseconds on refill events, but their current count/time share is unknown. A meaningful redesign must reduce actual executed padding or share weight work without changing every real prompt's native layers/experts. Before coding, record actual active/padded decode rows, prompt lengths/capacity/span, per-mixed GPU spans and first-frame handoff. Prior span/compact/layout wins already applied; they are not new candidates.
3. **Keep more useful model work between host completion barriers.** Current complete_decode reads semantic tokens to CPU after each one/two-frame graph, and first-frame finish also reads output. Fixed scripted replay can isolate graph-only submission from host-control cadence. A device-resident completion/admission loop may help only if exposed inter-graph holes are measured at millisecond scale. Current client logs do not establish such holes; replacing Python with Rust alone is not a measured remedy. Existing bounded pipeline improves kernel overlap only~0.393ms per two-frameB64, so broadening its remaining eligibility cannot supply4ms per full64 frame equivalent.

No currently measured structural candidate has a proven6.47s service budget. The proposed causal replay may reveal one; failure to find it is not proof of impossibility.

## Minimal diagnostic and scope

Reuse frozen e7cda1e and the passed80-request/414-entry dynamic trace. Hold initial model/request/codec/RNG state, actual generated frames, slots/resets, codec input rows and original DP cost table/grouping fixed. One untimed correctness replay records the sequence; timed work replays do not perform per-step CPU output hashing. Compare model-only, codec-only normal/cached, serialized normal/cached and overlapped normal/cached. Preserve exact model outputs/metadata/final states and per-codec-call PCM/final codec state; skipped subsystem state must stay initial. Record wall, device-stage unions/overlap, actual roster/padding and CPU enqueue time. Resolve timing events after final join and compare one unobserved run to quantify observer overhead.

This80-request fixture uses packed prefill and single-step DecodeGraph; it contains neither current mixed prefill nor frame pipeline. It can identify codec/model interference and graph-submit exposure under fixed native work, but cannot establish the current full-service mixed fraction or target feasibility. If it reveals material interference, the next minimum current-service profile is one nonblocking timeline of the existing full1088 configuration with explicit benchmark phase boundaries, active/padded batch, prompt/capacity/span, codec groups/DP table, queued/pending/runnable counts, main/codec events and CPU submission/readback scopes. Keep that profile diagnostic-only and separately compare unprofiled fixed work before attributing a speedup.

Evidence: [reproducible CPU ledger script](audit.py), [all four delivery ledgers and input hashes](current-delivery-ledgers.json), [cached service result](../codec-cache-service-summary.json), [admission service result](../pipeline-admission-service-summary.json). All outputs and tails retained.
