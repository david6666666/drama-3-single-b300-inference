# Bounded MoE counter diagnosis after metadata-ready screening

Status: CPU plan and offline metric support checked; GPU sampling has not started. The parent authorized preparing/fixing the new artifact helper and protocol, then the first eight node records after reporting frozen hashes and the final metric list. Whole-graph collection remains a separate decision. No runtime source or completed experiment is edited.

The next question is whether current native versus metadata-ready execution loses time in weight supply, insufficient resident work, or dependency scheduling. The previous exact candidate did not pass the 5% continuation gate. Repeating a tile or register sweep without this evidence is not proposed.

## Evidence already reusable

The archived native sample (`../../drama3-mixed-20261016/ncu/summary.json`, source7a8cea1) collected six B64 GEMMs in one profiler pass each. FC1/FC2 DRAM reads were1.00052–1.00138 times logical active-expert weight bytes, with56.36–75.54% peak DRAM throughput and3.46–4.48% tensor-pipe activity. This supports memory-supply analysis; it does not establish repeated full-weight fetches as the waste. Its204/163/116 expert counts were recomputed with torch.topk, not captured native IDs. Current actual native IDs yield206/159/115. Old isolated kernel times cannot be subtracted from new full-call times.

The archived wide CSV also already reports native512-thread blocks, REG128,149,504 allocated shared bytes, launch occupancy limits of one block from registers/shared memory, and15.57/31.14 static waves per SM for FC1/FC2. These are launch/theoretical fields. The grid is based on worst-case CTA288; invalid work and persistent scheduling mean these are not measured useful-wave counts or tail time. `sm__cycles_active` is cycles with any warp in flight, not achieved occupancy. Old data contain no L2 sectors, actual CTA residency, eligible warps, stall reasons or current candidate counters.

The current M128 candidate has exact native FP8 boundaries, top8, BF16 FC2 and original ordered finalization. It has four operand stages versus native eight; FC1 performs16 K128 iterations and FC2 four. Its global persistent scheduling, actual384-thread blocks and roughly74.6KiB shared allocation are recorded in frozen CUBIN/resource evidence. PTX register hints do not prove real register allocations; the actual TS CUBIN has REG32 and no emitted SETMAXNREG. Neither stages nor static resource differences identify a bottleneck by themselves.

## Fixed first collection

Use only original L0/real and L39/real fixtures under source95fcb0e. They cover relatively dispersed206-expert/208-token-tile and concentrated115-expert/146-token-tile native routing. Both remain the original physicalB64 fixture (63 live plus original padding). Each arm uses its independent live metadata/workspaces and full native math. No expert pruning, new quantization, clock changes, cache flushing, altered PDL flags, routing or tactic sweep.

Four new processes, each executing the complete graph and selecting exactly its two FC kernels: L0/native, L0/candidate, L39/native, L39/candidate. This yields exactly eight node records. The baseline remains native prepared tactic(8,5), PDL=True; the candidate remains the frozen metadata-ready M128 chain, PDL=True. All metadata producers, activation and finalizer execute even though only FC1/FC2 are selected for counters. Candidate graph retains metadata→FC2 DEFAULT and the original PROGRAMMATIC chain; retain graph/IO/weights/descriptors throughout the run.

Warmup and the existing exact logical-map/ownership checks occur outside the selected profiler scope. Then one complete graph replay is bracketed by profiler start/stop. The driver checks exact logical intermediates/finalized outputs, unchanged inputs/weights/RNG/guards, current independent maps, graph identity and source/CUBIN hashes afterward. This diagnostic does not repeat all15 accuracy cases already passed, nor claim a new timing trial. Each output directory is new and refuses overwrite.

Use explicit `--config-file off --target-processes application-only --profile-from-start off --replay-mode kernel --graph-profiling node --launch-count 2 --cache-control none --clock-control none --pipeline-boost-state dynamic`, exact FC kernel filtering and the fixed list in `metrics.json`. Five graph warmups match the prior screen. Native and candidate retain their own legal graph layout and allocation; no eager stand-in is allowed.

The17 requested metrics are all present with their exact suffixes in the installed NCU2026.3.0 offline gb110 catalog, queried with `CUDA_VISIBLE_DEVICES=''`; this proves support naming, not collection success or pass count:

| Purpose | Metrics | Interpretation |
|---|---|---|
| Time/DRAM | duration; DRAM read/write bytes; DRAM peak% | Actual current serialized bytes and bandwidth; not service time |
| L2 | read/write sectors; overall sector hit rate | Convert sectors to32bytes; combined hit rate is not weight-only hit rate |
| Resident work | CTA active per active cycle; warp active% | Actual average residency; not just launch occupancy limit |
| Issue supply | eligible warps per active cycle; issue-active% | Distinguish many resident but ineligible warps from insufficient resident work |
| Waits | barrier, long scoreboard, short scoreboard, membar per issue-active ratios | Directional aggregate waiting evidence; not disjoint wall-time percentages |
| Existing context | SM cycles active%; tensor-pipe active% | Maintain old diagnostic context, without equating activity to occupancy |

Record raw counters, units, profiler passes, kernel names, actual grid/block/shared/register launch fields and all n/a values. No `--set full`, PC/SASS instrumentation or automatic metric substitutions. Per-process watchdog600s, maximum four processes; if any kernel needs more than four replay passes, finish/preserve that observation and stop before another process pending review. Any unsupported requested metric, unexpected selected kernel count, first semantic/hash failure, graph mutation, ownership conflict or timeout stops the series. Poll designated GPU ownership every2s, accept only this launch process group, and only terminate our own group. These samples are finite observations, not continuous proof of exclusivity.

## What the first collection cannot establish

Kernel replay with node profiling serializes work and may alter cache history even with cache-control none. Its durations cannot be summed to infer actual PDL overlap, its final wave cannot be inferred from average stalls, and its bandwidth is not the complete graph's achieved bandwidth. NCU's native/typed-PDL internal transformation is not a formal substitute for the graph dependency proof. Preserve pre/post snapshots and exact outputs, but report this limitation regardless of PASS.

The independent same-version audit is `../ncu-metadata-ready-replay-audit.md`. NVIDIA documents node serialization and whole-graph profiling in the [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay). Range replay does not support Graph Management, so it is not an automatic fallback for the typed graph. App-range has separate warmed/uploaded-graph limitations and is not proposed for this first collection.

Actual phase-tail/TMA-stall attribution remains missing. The source proves8 versus4 stages, but aggregate barrier/scoreboard metrics cannot label a particular TMA stage as starved. If node records identify a specific unresolved memory/occupancy question, propose at most four `kernel + graph` aggregate records covering exactly metadata+FC1+activation+FC2+finalizer, using only graph-supported hardware metrics. Obtain parent review first. That mode can preserve inter-node interactions but does not provide per-node attribution; node-level and graph-level times are different observations. A separate bounded four-call CUPTI activity timeline could establish start/end overlap and final-kernel tails, while PC sampling at one implicated kernel could identify wait PCs; neither is authorized now and neither is needed by default. A detailed stage cause requires a specific PC/source/trace correspondence, not speculation from the source's stage count.

## Recomputable budget and decisions

`cpu-budget.json` records formulas and source values. Current unprofiled single-screen complete calls are120.748→117.452µs (L0) and79.944→76.630µs (L39), only2.729% and4.145%. No twelve-stress performance or three-repetition result is implied. A new10% complete-MoE hypothesis must cover12.075µs at L0 and7.994µs at L39 relative to native; from the current candidate it must remove another8.779µs and4.680µs, respectively. Clearing5% instead still needs2.742µs and0.683µs beyond the present candidate in this one screen. These are target gaps, not measured recoverable waste.

For historical context only, applying90% of the old run's reported implied peak to that same run's total measured bytes would reduce its FC times by16.1–37.4% (roughly10.4–12.6µs each). This is a conditional throughput scenario, not current headroom: cache, graph overlap, scale/activation traffic, frequency, kernel setup and structural overhead all matter. Do not turn it into a40-layer or service projection. Unique logical current weight volumes648,019,968B (L0) and361,758,720B (L39) likewise are not obligatory HBM traffic because warm L2 may retain weights.

After current counters, compute for each isolated kernel `B = DRAM_read + DRAM_write`, `BW = B / measured_time` and conditional `max(0, measured_time - B / chosen_bandwidth)`. Quote chosen bandwidth and provenance; no assumed90% achievement. This gives a memory-transfer scenario, not additive critical-path savings. Duplicate-L2 traffic, residency and wait ratios must explain why a concrete change could approach it. Only a complete-call exact+screen can validate the net result.

| Observation | Next decision |
|---|---|
| Current DRAM close to one logical weight copy; higher L2 demand but no meaningful DRAM excess | Do not build another weight-reuse kernel merely to remove global buffers; inspect supply/occupancy |
| Candidate changes actual residency/eligible work and current bandwidth enough to cover≥10% complete-call budget under an explicit conservative scenario | Propose one scheduling change tied to those measured counters, including its full-call overhead |
| Lower bandwidth with substantial barrier/scoreboard and few eligible warps | This identifies a dependency candidate, not TMA causality; request one targeted source/PC/timeline check before another kernel |
| Full-grid theoretical waves differ from active token tiles | Keep invalid/persistent work accounting separate; do not estimate tail solely from launch waves |
| Isolated improvement exists but full-call screen remains small | Seek whole-graph evidence for overlap/activation/finalization cost before changing GEMM stages |
| No counter-supported substantial local budget | Stop this local hypothesis and prioritize other native service structures; the user's overall throughput target is unchanged |

All original negative results remain frozen. This diagnostic is an attempt to identify a larger native improvement; it is neither a promotion nor a reason to lower the user goal.
