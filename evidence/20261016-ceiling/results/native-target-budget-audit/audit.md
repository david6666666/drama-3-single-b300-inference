# Native execution budget against the C64 target

Status: CPU/source audit only, 2026-10-16. No new GPU execution, runtime change, or performance promotion. All earlier negative gates and the PC-sampling INCONCLUSIVE result remain unchanged. `derive.py` reproduces `budget.json`; that JSON records the exact input hashes. The goal remains 6765.407263 frames/s at C64 and the previously agreed quality bounds, followed by all-concurrency validation.

## The gap is in sustained model execution

The e7cda1e diagnostic completed all 1088 requests and delivered 105160 frames in 22.897230 s (4592.695 frames/s). At fixed delivered work, the target takes 15.543780 s: 7.353450 s, or 32.115% of this run, must disappear. The best listed unprofiled single arm is 4726.771 frames/s, with its own 101529-frame workload; its gap is 6.472487 s. Different output lengths prevent treating the difference between those runs as a controlled speedup.

| Diagnostic interval accounting | Seconds | Interpretation |
|---|---:|---|
| Pure decode union | 15.803838 | 1035 graph calls, 1850 native frame steps |
| Mixed backbone plus teacher | 4.946357 | 196 admissions, 1079 new prompts |
| All model union | 21.289383 | Includes first frames, old tower and initial prefill |
| Codec union | 6.679043 | Work on its own stream; mostly overlaps model |
| Model/codec intersection | 5.709033 | Count once, not twice |
| Codec only | 0.970010 | Outside the model union |
| Neither model nor codec | 0.613251 | Inside the measured service envelope |

The interval envelope is 22.872643 s, slightly shorter than the client wall time. Deleting every codec-only and neither interval would save only 1.583260 s, and those intervals are not all removable. Deleting the entire current mixed interval, an intentionally impossible screening assumption, still gives only 5858.211 frames/s on this fixed trajectory. These calculations identify coverage needed; they do not prove the goal impossible. Changing the execution strategy also changes the trajectory.

The CPU `first_frame` scope contains about 5.106 s of waiting for work already present in GPU intervals. It is not another independent five-second scheduling budget. The latest timeline records step counts, so no old synchronous 384-request percentages were applied to this asynchronous 1088-request run.

## What vLLM does well, and what is already present here

The inspected vLLM-Omni source is frozen at d6ed4f4844fbc41573ffc03247e5b2ff0a38d322. In `vllm_omni/deploy/fish_omni.yaml`, lines 6–7 enable asynchronous chunks and chunked prefill; 23–29 group admissions in sixes or after 100 ms under load; 33–37 enable codec graphs and stateful convolution/KV; 49–51 enable asynchronous scheduling and a 4096-token budget; 56 selects BF16 GDN state; 59–66 combine full decode graphs with piecewise mixed graphs; 85 overlaps waveform delivery; 98–100 enable MPS. These are mechanisms, not evidence that a larger configured maximum creates more than 64 ready requests at C64. The config's codec FP16 at line 30 is not silently adopted into the current native BF16 codec contract.

The independent engine now already has a shared mixed backbone, packed prefill, continuous admissions, reserved-padding routing, graph replay, frame pipelining, BF16 state, and asynchronous stateful codec/delivery. Porting those names a second time is not a new source of savings. The archived upstream 256-request continuous trace averages 60.718 pure-decode rows during intake; the current engine's intake decode time-weighted active count is 60.252. These are different workloads and are comparable only as occupancy descriptions, not timings.

For the current engine, the optimistic assumption that every intake step costs the same after filling to 64 yields only 6.22% more row throughput, or 0.833233 s of the intake pure-decode time. It is neither an attainable saving nor a hardware lower bound. The final drain is 1.669578 s and delivers 4312 real frames; it cannot simply be deleted or added again to a batch-fill saving. The estimate of extra executed rows is only 0.375%, conditional on the archived EOS/max-frame accounting. Actual finish reasons would be needed to make that last number exact.

## Remaining native strategies and their coverage

| Strategy | Prior coverage or conditional budget | Required constraints and smallest useful validation |
|---|---|---|
| Reuse identical prompt rows through stateless functions | In the FIFO proxy for current mixed admissions, 64585 repeated prompt rows; retaining current 256-row padding removes 17.41% of mixed capacity, versus 26.37% with 64-row padding. Extreme linear scaling of the entire mixed interval gives 0.796–1.258 s, not a forecast. | Reuse only bit-identical inputs, preserve every FP8/BF16 boundary, expand outputs to original causal rows before GDN/attention/tower state. All-layer equality, then complete mixed graph timing including compact/expand. Actual admission IDs remain missing; current grouping is a proxy. |
| Reuse recurrent shared prefixes as a causal DAG | Could cover work beyond stateless MoE/dense functions, but existing data do not establish a separate additive budget. | Preserve FP32 internal GDN checkpoint, convolution history, attention KV and teacher lookahead; copy/branch state at a proven identical prefix. Do not claim a simple prefix pass plus suffix pass is free: it rereads forty layers of weights. |
| Publish old hidden rows before new teacher suffix | Historical fixed mixed graph teacher suffix averages 4.55149 ms; scaling 196 admissions gives a gross 0.892092 s proxy. Old tower alone is 0.261233 s in current service. | New teacher and old slots may run independently, but sampling must reserve distinct original step values, private attention/workspaces, and join before committing global state. Compare the complete mixed→next-step DAG, including the new small batch that must catch up; never time only the overlap window. |
| Native expert-kernel supply/pipeline improvement | Historical pure B64 profile attributes about 3.999 ms to forty FC1+FC2 pairs; multiplying by 1850 steps covers about 7.40 s, but is not a measured allocation of the latest service time. | Preserve original weights/scales, all eight experts, MMA accumulation order and every native quantization/finalization boundary. New changes must target measured traffic or a proven dependency; complete native MoE timing is the local gate. Existing tiles, partitions, resident chain, Lt, priority and route-row cache failed their gates and are not repeated. |
| Truly ragged native codec execution across frame buckets | Current codec uses 2206 calls for 2064 actual frame groups and 972 emits; total valid→bucket→batch padding is 16.40%. A hypothetical 25% codec reduction covers 1.670 s of codec events but only 0.2425 s of proportionally attributed codec-exclusive time. | Keep actual frames, per-slot FIFO, history/KV and native precision. A segmented decoder differs from previously failed padded coalescing; it needs complete PCM/state checks and full service measurement. The observed 1.394 s codec-overlap association is not causal recoverable time. |
| Hardware transparent lossless weight allocation | Newly proposed; no measured compression ratio or saving yet. This could change actual DRAM bytes without changing model arithmetic, unlike a scheduling-only tactic. | Verify the actual allocation compression flag, preserve all stored FP8 bytes and scales, reuse the original native kernels, then test native complete MoE and actual DRAM traffic. CPU entropy/zlib is only motivation. |

The row-reuse route is already under independent investigation; this audit does not launch a competing implementation. The historical teacher proxy uses a different source and fixed D58/P6 shape, so the current per-admission teacher time is still missing. The teacher, row-reuse, padding and codec budgets overlap and must not be summed as achieved or simultaneously available savings.

## Why extra queues do not automatically reuse expert weights

Different backbone layers have different weights. A request's next true frame input requires the previous frame's forty backbone layers and tower through codebook 1; the existing frame pipeline uses that earliest dependency. C64 therefore supplies at most 64 ready pure-decode tokens, not another batch of speculative future tokens. Prefill is the main supply of additional known tokens and already shares the mixed backbone weight path.

Splitting ready rows into queues at different layers can turn one B64 weight traversal into two B32 traversals. It can help only with evidence of ready-row aggregation or real cache residence, not by assuming weights survive the other 39 layers. One expert's original FC1+FC2 weights occupy 3 MiB; all 256 experts are 805.306 MB per layer and 32.212 GB across forty layers. Archived real B58 and B63 steps touch 6064 and 6206 distinct expert-layer pairs, corresponding to 19.076 and 19.522 GB of one-read logical FP8 weight bytes. These examples are not the service average or an actual DRAM lower bound.

The new native/candidate NCU node records show approximately one active-weight traversal, not a hidden 2× reread to eliminate. The candidate reduces some serialized kernel times without materially reducing bytes. Node replay serializes PDL and cannot measure the complete-call overlap budget. Long-scoreboard ratios are sampled issue/stall statistics, not percentages of wall time or proof of TMA weight starvation. The subsequent PC-sampling attempt remains INCONCLUSIVE and supplies no causal attribution.

## What is still needed to prove a feasible hardware budget

The isolated 8.275 ms B64 decode is an achieved reference under its own conditions, not an upper bound or a service lower bound. The following are not yet measured together: the current service's per-layer route unions and physical weight reads, cache reuse, dense/GDN/KV traffic, codec contention, sustained combined bandwidth and critical-path tail costs. Only after obtaining unavoidable physical bytes and sustainable bandwidth for the same workload could `bytes / bandwidth` support a memory lower bound. Existing data do not establish that 6765 frames/s is achievable or impossible.

As a deliberately conditional screen, taking historical 68–76% peak bandwidth at face value and allowing its FC work to reach 100% could remove only roughly 24–32% of that FC duration (about 0.96–1.28 ms per historical B64 step). That is not a prediction, ignores latency and other bottlenecks, and cannot be applied as an exact service saving. A substantial physical-byte reduction, better simultaneous use of independent work, and/or broader prompt work reuse is still needed to explain the whole 6.47–7.35 s gap. This motivates the native compressible-allocation investigation rather than relaxing the target.

## Reproduction and evidence limits

Run `python3 derive.py` in this directory. It imports only the Python standard library and reads the archived JSON/Chrome trace files. `budget.json` stores hashes for all inputs. Source anchors are pinned above; no live service results were created. Other supporting artifacts are `cpu-service-ceiling`, `native-row-reuse-audit`, `expert-m128-ncu-node-v2`, `expert-m128-fc1-pcsampling`, `codec-kernel-profile`, and the upstream continuous schedule JSON. All reported opportunities are hypotheses with distinct gates, not additions to achieved throughput.
