# L0 FC1 PC sampling: INCONCLUSIVE

The bounded run stopped after the native record. The original metadata-ready candidate was not sampled; no repetition or parameter change was made. The native complete graph passed all 80 exact checks, including retained graph/owner state checks. This is neither a candidate performance failure nor evidence that an ordinary load causes the prior long-scoreboard increase.

The report contains 4,815 samples, including 3,503 long-scoreboard samples, with a 4,096-cycle automatic interval and zero reported buffer overflow or dropped bytes. Sample count exceeds the predeclared descriptive threshold. However, the PC correlation IDs are loaded addresses while the frozen disassembly uses function-relative offsets. All 445 sampled addresses therefore fail the conservative direct-offset gate; 424 return a SASS instruction and 21 return no instruction through the report API. No module base was inferred from a minimum PC or instruction-text similarity.

The NCU log and `profiler__replayer_passes` both report one pass. `smsp__pcsamp_aggregated_passes` reports two. The installed 2026.3 profiling guide describes the latter as the passes required for statistical warp sampling, but does not resolve this discrepancy. The original report and stop remain intact. CPU-only inspection of report-owned function metadata can clarify whether an exact address mapping is recoverable; it does not authorize another launch or change this recorded outcome.

All 147 frozen inputs still match. The designated GPU was empty before and after the process; nine sampled process audits observed no foreign context. GPU2 is released. No runtime code changed and no service gain is claimed.

The complete predecessors and PDL settings were preserved. Node-level NCU replay serializes the selected workload; PC counts cannot be converted to a service-time saving or added to another kernel timing. The unexecuted candidate prevents a native/candidate PC comparison.

Evidence: `protocol.json`, `frozen-inputs.json`, `launch-result.json`, `L0-native/result.json`, `L0-native/graph.json`, `L0-native-ncu.ncu-repz`, `L0-native-pc-result.json`, and `L0-native-process-audit.json`.
