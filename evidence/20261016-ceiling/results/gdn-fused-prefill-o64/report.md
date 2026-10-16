# Native GDN H/O fusion with original O V64 tile

This separate artifact follows the V32 fusion's exact-but-slower result. It preserves the original O V64 tile, using8 warps and combining two H V32 slices into V64. The H K64+K64 and O BK128 arithmetic/rounding boundaries remain explicit. No runtime or installed package is modified. The motivation is source-backed: V32 duplicated O's QK matrix twice as often as the native V64 path. This is one structural follow-up, not a tile sweep.

All57 synthetic,522 actual layer/fullgraph and1026 timed-trajectory checks pass bit-exact. Real coverage is three mixed shapes×30 GDN layers and every captured output,111 mutable state tensors and RNG. Timing uses three alternating A/B repetitions per shape,20 consecutive graph replays per measurement, same initial state restored outside events, on GPU0 and frozen e7cda1e. Pre/post process checks were clear and task-owned memory was released; no continuous process audit establishes full-interval exclusivity.

|Mixed shape|Native median ms|Fused median ms|Time change|
|---|---:|---:|---:|
|d58p6|26.355219|26.420694|+0.248%|
|d40p3|18.257458|18.255545|-0.010%|
|d12p1|15.431131|15.429988|-0.007%|

The primary D58/P6 shape is still slower; the other differences are effectively flat and cannot support a useful improvement claim. The predeclared all-shape local continuation condition fails. Both H/O fusion candidates remain artifact-only; no service run is justified by these results.

The candidate compiles at206 registers/thread, zero spill/stack,132884B dynamic plus1024B static shared memory, and512 TMEM columns. This allows only one CTA per SM by shared/TMEM capacity. V32 had245 registers,115492B dynamic+1024B static shared,256 TMEM columns. Eliminating duplicate QK work did not create a useful complete-graph gain. The source also serializes O with each H chunk, where native O can parallelize all chunks after H. These are structural costs, not measured hardware stall attribution; a new profile would be required to apportion them.

A CPU/theoretical count of eliminated H/updated-V traffic is therefore not a measured DRAM saving or guaranteed time budget. Native intermediate cache residency, added live values, CTA residency and chunk parallelism can offset fusion. The full model still performs its original dense/MoE/attention/teacher work. No claim is made about end-to-end service, WER/SIM, or the unchanged6765.407frames/s target.

Protocol, CPU compile, exact checks, raw timings and cleanup are preserved. The first V32 experiment and all original failures elsewhere remain unchanged. No more tuning sweep follows this result.
