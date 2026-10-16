# Native GDN H/O fusion: exact, but no local performance gain

This artifact-only implementation fuses FLA's per-sequence H recurrence and O into one Triton kernel. It preserves BF16 H and updated-V rounding before O, original H K64+K64 dot/update order, O BK128 dot order, original exponential functions, and FP32 recurrence across chunks. H uses V32; this first candidate consequently changes O V64 to V32. The baseline is the original compact FLA frontend and native H/O pair on frozen e7cda1e. No model calculation, expert count, precision or user quality bound changes.

57 synthetic checks pass across partial/empty/nonzero-initial-state/sentinel inputs and captured replay. 522 real checks cover90 same-input GDN layers (three actual mixed shapes) and all complete mixed outputs,111 mutable state tensors and RNG. Three alternating A/B repetitions per shape then compare20 consecutive graph replays, with identical initial states restored outside the timed interval. All1026 timed-trajectory output/state/RNG checks remain bit exact.

|Mixed shape|Native median ms|Fused median ms|Time change|
|---|---:|---:|---:|
|d58p6|26.355579|26.490366|+0.511%|
|d40p3|18.262636|18.286597|+0.131%|
|d12p1|15.425816|15.449295|+0.152%|

All three medians are slightly slower. The candidate fails its predeclared continuation condition and is not installed/promoted. These are complete isolated MixedGraph durations, excluding codec, sampling, Engine and HTTP; no service gain or new quality score is claimed. Native output lengths cannot explain this fixed-graph result.

The production compile has245 registers/thread, zero spill/stack,115492B dynamic plus1024B static shared memory and256 TMEM columns. Debug stores require119572B dynamic shared; its capacity must not substitute for production. Runtime pointer/stride specializations have separately recorded cubin hashes; the CPU AOT and GPU specialization are not asserted byte-identical.

Source inspection identifies an avoidable cost in this first design: native O has two V64 tiles per head/chunk, while fused V32 has four and therefore duplicates QK matrix calculation twice as often. Native H uses four V32 tiles. A separate, explicitly declared O64 candidate retains O V64 and combines two H V32 output tiles with8 warps, keeping both K64 arithmetic blocks. CPU AOT reports206 registers, zero spill,132884B dynamic+1024B static shared and512 TMEM columns. This reduces QK duplication but allows only one CTA per SM by TMEM/shared capacity, so benefit is not implied. Its correctness/timing results live in ../gdn-fused-prefill-o64; this original outcome remains unchanged.

No frozen runtime, venv package, shared cache or clocks were modified. GPU0 process checks were clear before and after each process, and task-owned GPU memory was released. These runs did not continuously poll competing processes, so full-interval exclusivity is not independently established. Protocols, original inputs' generation, commands, source hashes, all checks and raw repetitions are retained beside this report. Full1088 WER/SIM and all-concurrency performance gates remain separate and unmet.
