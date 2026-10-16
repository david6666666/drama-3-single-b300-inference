# NCU replay audit for metadata-ready MoE

CPU-only audit, 2026-10-16. No GPU query, kernel launch, profile, or experiment-source edit was performed. Installed `/usr/local/cuda/bin/ncu --version` reports 2026.3.0.0 (build38493568); the launcher selects `/opt/nvidia/nsight-compute/2026.3.0`. Its local docs report13.4. Full local CLI help, version output and document hashes are retained in `ncu-metadata-ready-replay-audit/`.

Official behavior relevant here: ordinary per-kernel collection serializes launches. Range replay preserves concurrency but its supported-API list excludes Graph Management. App-range reruns the application, preserving workload interactions and aggregating the selected range; its instruction-level SASS data excludes JIT kernels. Whole-graph profiling treats the graph as one workload, preserving inter-node cache behavior; unit-level and instruction-level source metrics are unavailable, while hardware counters, some launch metrics and sampling remain supported. CUDA-event/host timings under NCU include observer overhead and are not standalone performance evidence. These restrictions are present in both installed and online [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay).

The installed help confirms four replay choices and default `kernel`, plus graph modes `node` (default) and `graph`. It also confirms defaults `cache-control=all` and `clock-control=boost`; both must be explicitly overridden to `none` for this study. Use `--config-file off` to avoid inherited CLI defaults and record the selected tensor pipeline boost policy. [Official CLI options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options)

The official 2025.4.1 notes, also shipped locally, warn that app-range node profiling rejects graphs uploaded outside its range. Therefore warmed/pre-uploaded graphs cannot silently be moved to app-range with default node mode. These notes also describe graph edge-data visibility; they do not constitute a formal guarantee that all profiler-internal PDL transformations preserve the application's typed-edge proof. [NVIDIA release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#updates-in-2025-4-1)

Recommended bounded protocol (our proposal):

1. Collect exactly eight node records: L0/L39 × native/candidate × FC1/FC2. Four driver processes can each produce two filtered records. Use the existing complete graph/metadata and PDL flags, with warmup and correctness preparation outside the selected profile scope; match the exact FC1/FC2 kernel names and require the expected count. Configure `--replay-mode kernel --graph-profiling node --launch-count 2`. Keep every predecessor in the executed graph even though only FC1/FC2 are selected. Do not disable PDL or omit producer work to make profiling convenient. Bound runtime and preserve any unsupported/deadlock result.
2. Interpret these as isolated, serialized counter observations. They can identify changed traffic or execution efficiency. They cannot measure launch overlap, assign a hidden PDL wait budget, or establish the full-body speedup. Even with cache-control none, per-kernel multi-pass replays can see changed cache history. Record actual pass count.
3. If the node observations support a concrete hypothesis, collect at most four whole-graph records for the same layers/arms, using `--replay-mode kernel --graph-profiling graph --launch-count 1`. Include the same scope in both arms: all metadata producers, FC1, native activation, FC2, and native finalizer. Check the sealed typed-edge snapshot before launch and graph outputs/ownership afterward. Graph-node sums must not be compared with whole-graph elapsed time as though they represented the same execution.
4. Use app-range only as an independently declared fallback when necessary, not an automatic retry. Its range is an aggregate workload, not a new source of individual FC1/FC2 records. Its upload restriction and exact CLI behavior with the existing warmed graph must be resolved before authorizing it. Do not replace the typed graph by eager serial calls, or use range replay over unsupported graph calls, and label that result equivalent.

For both forms use `--config-file off --target-processes application-only --profile-from-start off --cache-control none --clock-control none`, with the recorded CUDA context current before driver profiler-start. No clock reset, cache flush, driver permission change, or process interference is part of the proposal. Only the designated idle GPU is eligible; other contexts can contaminate device-memory counters. The existing unprofiled complete-body benchmark remains the performance reference.

Minimal six hardware metrics, already collected successfully on this B300 in the archived native-B64 counter evidence:

- `gpu__time_duration.sum`
- `dram__bytes_read.sum`
- `dram__bytes_write.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`

This is a proposed fixed metric list, not a claim that graph-level collection has already succeeded. Preserve unavailable metrics explicitly and stop for the owner to assess; do not silently change the list or enable the full section set. Keep resources from the already frozen CUBIN/AOT audit; no new occupancy sweep is needed. Instruction-patching/SASS statistics, warp sampling and PM timelines are deliberately outside the first bounded counter pass.

Source for already successful metric names: `source-moe-residual-6ff8d80/evidence/20261016-mixed/results/ncu/real-moe-metrics-base.csv`. No new numeric evidence or speedup is claimed. SAFE_TO_ARCHIVE for this report and its independent evidence directory.
