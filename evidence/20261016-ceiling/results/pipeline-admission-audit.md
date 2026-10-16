# Queued versus runnable admission: CPU audit only

Frozen source reviewed: a7118fb7eb748d08c3b774fa6c9189b6eab0d9cc, engine.py:436-451 and 532-571; frame_pipeline.py:6-14. No runtime edits or GPU work in this audit.

## What the measured counters prove

The complete candidate produced eligible159, capacity193, admission1216, audio_boundary142. Counters cover64 warmup plus1088 measured requests and count ordinary decode choices, excluding mixed/prefill branches. The present admission counter conflates `bool(waiting)` and `not pending.empty()`. It does NOT prove all1216 were waiting-only; that distinction was not instrumented.

The source makes an avoidable restriction plausible. Each ordinary iteration has already pulled pending requests into waiting and evaluated admission. A waiting request is admitted only when active<32, waiting>=6, or its queued age>=100ms (subject to free slots). With active>=32 and1-5 waiting requests younger than100ms, ordinary decoding continues deliberately while the engine waits to form a prefill batch. A successful mixed prefill branch continues the outer loop before ordinary decode selection. The pipeline selector subsequently rejects *any* waiting item, including this deliberately deferred and not-yet-runnable group. Remaining pending requests can also arrive between the first queue drain and selection; they are a separate condition.

The per-request phases are correlated: actual prior ledger had only790/1507 calls with capacity>=16 provably blocked by a scheduled chunk boundary, leaving47.58% as an upper bound before unknown admission state. In the new candidate, after capacity/admission rejection,159/(159+142)=52.82% selected pairs. Neither figure estimates waiting-only opportunities.

## Recommended bounded predicate

Keep the two-frame graph and all EOS/append semantics unchanged. Preserve the all-request `min(next_emit,max_frames)-frame_count>=2` horizon. At ordinary selection, use a fresh time and the post-cancellation live roster:

```python
pending_present = not pending.empty()
waiting_present = bool(waiting)
waiting_age = now - waiting[0].queued_at if waiting_present else 0.0
waiting_runnable = bool(free) and waiting_present and (
    len(active) < admission_min_running
    or len(waiting) >= admission_min_batch
    or waiting_age >= admission_max_wait
)

allow_pair = (
    supported_capacity
    and not pending_present
    and not waiting_runnable
    and all(min(r.next_emit, r.max_frames) - r.frame_count >= 2 for r in active)
)
```

Only relax rejection for waiting requests which the existing admission policy intentionally defers. Continue rejecting nonempty pending conservatively; do not infer pending eligibility from approximate qsize, drain it a second time, change batch6/maxwait100ms, or add admission during a pair. Re-evaluate the admission predicate because elapsed time or cancellation can make a formerly deferred group runnable after the earlier decision. Use the request's original queued_at, matching the current source, not its transfer time into waiting.

If a waiting deadline expires while the pair runs, or a new pending request/cancellation arrives just after selection, it is observed at the next ordinary scheduling boundary. Relative to a single step, this can add at most ONE MODEL STEP before the next check; it is not a fixed millisecond or strict100ms wall-time guarantee under codec interference. Existing batching already checks deadlines only between GPU work. This tradeoff must remain explicit and TTFA must be measured. No predicted EOS, row removal, skipped experts, extra audio chunks or altered ramp is introduced.

## Instrumentation before interpreting another pair

Keep mutually exclusive decision counters: eligible, capacity, pending, waiting_runnable, audio_boundary. Add a separate context object, so diagnostic fields are not accidentally included in the decision denominator:

- waiting-only, pending-only, both-present, neither counts at supported-capacity selections;
- eligible_with_deferred_waiting and eligible_without_waiting;
- deferred waiting size histogram1..8;
- waiting-runnable threshold reasons (active count, batch size, expired deadline), clearly marked nonexclusive;
- oldest waiting age sum/max for selected deferred pairs, plus count of selections near the deadline if a documented reporting bucket is useful.

Log this once at shutdown; no per-layer/step GPU timing or synchronization. Existing lifetime counters include warmup; keep that label unless a separately authorized request-phase marker is added. These counters reveal how much of1216 is actually recoverable; present data cannot.

## Validation and impact limits

CPU tests should cover active31/32, waiting5/6, age just below/exactly100ms, empty/nonempty pending, no free slot, post-cancellation threshold crossing, and horizon1/2. The unchanged CUDA graph retains the3689-check evidence, but scheduler completion/selection tests need the new predicate. A new same-source full service pair is required before choosing the policy; score every output again because sampling_step/roster schedules can change. Native math and user quality limits remain unchanged.

The measured159 pairs could recover only159*0.393184=62.5ms of isolated model time, about0.27% of22.755s before overlap. More opportunities may help, but this direction alone is not evidence for bridging4580 to6765fps. Current default-off feasibility delivered only+0.0698% throughput and should not be promoted as a material service optimization.
