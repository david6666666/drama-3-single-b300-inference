# Compact packed GDN validation

Frozen source `1618ad4b2c7591465a47ca0dad8100e782820cd2`; GPU2 B300. Default-off `DRAMA3_COMPACT_GDN=1` is the only changed runtime variable. Native BF16 dense/GDN and full top-8 original FP8 expert weights/scales are unchanged.

All 2099 exact checks passed across 6 cases, including identical-input outputs and all state for 30 real GDN layers, complete graph outputs/state, untouched-slot ownership, zero-length sequences, partial chunks, mutable captured descriptors, reused slots, a real 264-token prompt requiring span512, and the parent PackedPrefillGraph path.

| Case | Static ms | Compact ms | Saved ms | Latency decrease |
|---|---:|---:|---:|---:|
| d58-p6 | 28.365 | 27.773 | 0.592 | 2.09% |
| d58-p6-descriptor-update | 28.316 | 27.714 | 0.601 | 2.12% |
| d40-p3 | 20.308 | 19.522 | 0.786 | 3.87% |
| d12-p1 | 17.244 | 16.370 | 0.874 | 5.07% |
| span512-fallback | 22.128 | 20.296 | 1.832 | 8.28% |
| parent-packed-d58-p6 | 26.998 | 26.435 | 0.563 | 2.09% |

These are medians of five alternating CUDA-event samples after one warmup, with identical initial state restored outside each event. The mixed graph includes the joint backbone and teacher-forced prefill tower; it excludes service scheduling and audio emission. These measurements are not a service throughput claim or a substitute for full quality scoring.

Actual baseline D64/P1280/span256 profile (`mixed-profile/table.txt`, `trace.json.gz`) is 28.322 ms. Relevant kernel totals are chunk output 0.978 ms, recomputation 0.535 ms and KKT solve 0.636 ms. The 0.967 ms recurrent-state kernel already iterates only actual sequence chunks, so it is not a removable masked-work budget. The measured ~0.592 ms graph saving is consistent with removing a portion of these costs, not halving all GDN work.

Evidence: `compact-gdn-validation.json`, `.log`, `compact-gdn-command.json`, `compact-gdn-experiment.json`, `validate_compact_gdn.py`, and the immutable `source-compact-1618ad4` worktree. Driver/source hashes and raw timing arrays are retained. GPU2 released to 4 MiB / 0% after PID 223542 exited.
