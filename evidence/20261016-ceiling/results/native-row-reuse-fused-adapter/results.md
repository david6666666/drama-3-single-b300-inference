# Full mixed inverse-aware adapter: exact and first performance gate passed

Frozen source e7cda1e remained clean. The original expanded-adapter failure is preserved in `../native-row-reuse-mixed`; this is a separate candidate with the same 5% criterion.

**603/603 exact checks passed** across all3 real cases: complete outputs, all111 model/request state tensors including nonparticipating slots, CPU/CUDA RNG, explicit mapping/compact slot contents and protected boundaries. The same captured graphs changed alias order/reference and slots. There was no numeric or harness failure.

| Case | Original mixed median(ms) | Candidate median(ms) | Saved(ms) | Latency reduction |
|---|---:|---:|---:|---:|
| proxy190 | 29.089808 | 27.272160 | 1.817649 | 6.2484% |
| proxy190_reversed_reused_graph | 29.093920 | 27.276113 | 1.817807 | 6.2481% |
| proxy192_changed_conditioning_reused_graph | 29.255775 | 27.281392 | 1.974383 | 6.7487% |

All3 cases exceed the predeclared5% local continuation gate. This is one A100→B100 screen percase, not repeated stability, service throughput or quality evidence. Each warmup and measured replay restores identical complete111-state snapshots before its start event; restoration is excluded equally. Actual compact gather, original fulltop8 MoE, inverse-aware native combine/RMSNorm, full backbone/GDN/attention, teacher tower and output gather remain inside each measured MixedGraph. Separate postgraph newfirst/olddecode sampling towers and codec are outside this scope.

The exact gate precedes timing; full states were not rechecked after each100 timed replay. Timing contains no observer copies. Fixed candidate buffers occupy 441087536 bytes; peak experiment allocation 159135906816 bytes includes full initial snapshots.

The adapter removes159 index_select launches and1,048,832,000 logical bytes of expansion read/write pergraph compared with the earlier adapter. These are source counts, not measured DRAM traffic or a causal attribution of the observed reduction.

No runtime integration or additional repetition was launched. FIFO proxy membership remains unverified against request-ID logs; no claim toward6765fps follows directly.

Protocol SHA256 `75477defba7953f5f22276c357f7ff088d592e74fc2fa54adf7bea708c5e0b57`; result SHA256 `8288b88b60d0d8758dc5fc14845cd989fc10802ebd91dc84d1f98850f3a51884`.
