# Current mixed native FI timing decision

Frozen1618, GPU2, D64/P1280/span256 with actual58:6 inputs. All later layout/conv-guard changes are absent from this source; compact GDN is off. Both graphs were captured before real prompt initialization; all conv/GDN/KV/request/RNG state was restored outside each CUDA-event interval. One warmup and five alternating samples per arm.

| Arm | Median ms | Raw five samples ms |
|---|---:|---|
| fla_ms | 28.368896 | 28.364799, 28.377089, 28.368896, 28.377056, 28.358656 |
| flashinfer_ms | 27.864992 | 27.858879, 27.867071, 27.869152, 27.854816, 27.864992 |

FI reduces this graph latency by0.503904ms (1.776%), below the predeclared >5% local reduction needed to justify another full1088 serving feasibility. Do not promote FI or start an additional service/quality run for this result. The original all-sequence numerical gate remains FAIL, and the separate diagnostic found final prefill hidden RRMS0.091184; these are not overwritten by timing. The independent reference and causal padding isolation are retained in `fi-mixed-reference-results.md`. Native full expert computation and precision were unchanged.

Compact FLA independently showed a similarly small graph saving while preserving exact outputs/state in its validation. These separate runs do not establish a tightly paired ranking between the two optimized candidates, but provide no compelling benefit to offset the unresolved FI quality work.

Evidence: `fi-mixed-timing-experiment.json`, `fi-mixed-timing-command.json`, `fi-mixed-timing.json`, `.log`, `time_fi_mixed.py`. Original failed report SHA256 remained unchanged: `2adae679605e2e1759a33ef6407ff1a2c74c20db9797e631c15cfa283e131620`. PID246552 exited and GPU2 returned to4MiB/0%.
