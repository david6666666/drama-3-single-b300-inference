# Native metadata-ready FC2: exact PASS, screening FAIL

The artifact-only candidate passed **1,516 exact checks** but failed the declared continuation threshold. All three real-layer complete-MoE reductions were below 5%. The launcher stopped after one AB screening; no three-round timing, stress performance, runtime integration, or service claim followed. The user's all-concurrency throughput target remains **INCONCLUSIVE / unmet**.

| Case | Native complete call (µs) | Candidate complete call (µs) | Reduction | ≥5% screen |
|---|---:|---:|---:|---|
| L0 / real | 120.748158 | 117.452478 | 2.729383% | FAIL |
| L19 / real | 99.425278 | 96.260157 | 3.183417% | FAIL |
| L39 / real | 79.943681 | 76.629758 | 4.145322% | FAIL |

These are one-round feasibility measurements: five warmup replays, 100 timed complete graph replays per arm, native A then candidate B, frozen GPU2. Native baseline remains original vendor tactic `(8,5)` with PDL enabled. Both include live native metadata generation, FC1, original activation/FP8 boundary, FC2 and original ordered top-8 finalization. Common input quantization and logits-to-top8 are excluded equally. Original physical B64 has 63 live rows and its original padding row. Results do not support a formal gain, a service gain, or a causal comparison with older experiments run at other times.

The new restricted FC2 permits its independent weight producer to load before activation completion. Its B and combined-scale tasks retain original dependency waits. FC1 retains its global metadata wait. All checkpoint FP8 bytes/scales, native quantization boundaries, BF16 outputs, all256 experts/top8 and M128 four-stage configuration remain unchanged. The installed generic validator is unchanged; only the explicit artifact typed contract accepts this schedule.

Actual full captures had five nodes: one native routing-metadata producer and four body kernels. The guard added one metadata→FC2 DEFAULT full-memory edge, preserving four original PROGRAMMATIC edges (including metadata→FC1) and the three body edges at ports1→0. Each native oracle used its own live mapping with original-token/expert ownership checks; eager and reused graph outputs, scales, guards, padding/tails, inputs, weights and RNG were exact. CPU/GPU CUBIN hashes matched. The extra same-graph lifecycle probe had six metadata-copy nodes plus two GEMMs; it passed zero→positive→zero with a fixed nonzero grid, restoring only ready activation references for the positive run. No empty native activation/finalizer semantics were assumed.

The gate ran all fifteen real/stress distributions; **stress performance was NOT_RUN**. The raw screening report records `stress_none_worse5pct: null`, avoiding a vacuous pass. The screening also completed 146 exact/read-only checks. Both owned child processes exited successfully and post-process GPU2 audits were empty. Audits sampled every two seconds and do not prove continuous exclusivity.

Frozen source is `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`. `protocol.json`, 130-file `frozen-inputs.json`, CPU AOT/PTX/SASS reports, guard tests, command logs and raw results are retained. All 130 frozen files still match after execution. FC1 CUBIN is `7c9bea320d547429ee935f7f43834afd931c0457dd2853cd277af1559984c4aa`; FC2 is `ea52321ec64fc9976a306b7c7aeeb0805033c2e024034e5cc6ce266e7d95b170`. Actual shared usage is 74,616 / 74,664 bytes, with no spill. No actual SETMAXNREG exists in the candidate CUBIN; PTX hints are not an occupancy claim.

Pre-GPU CPU mistakes are preserved separately: initial shape-field order, the initial incomplete configuration scope, a derived-default prefetch rejection test, and a repeated compile into a nonempty artifact dump. None is presented as a numerical kernel failure. `CPU-CONTRACT.md` remains the frozen pre-run document; these executed results supersede its pending-hardware statements. The earlier `expert-m128-pdl` negative result remains unchanged.
