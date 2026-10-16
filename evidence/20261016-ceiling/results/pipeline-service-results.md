# Bounded two-frame pipeline: full service feasibility

Same frozen source a7118fb7eb748d08c3b774fa6c9189b6eab0d9cc on GPU6, only FRAME_PIPELINE0→1; reserved router1 and five native flags are shared. One pair,1088 unfiltered requests/arm, no profiling.

| Arm | frames/s | TTFA p50 ms | TTFA p95 ms | RTF mean | WER % | SIM |
|---|---:|---:|---:|---:|---:|---:|
| control | 4576.835 | 137.852 | 266.791 | 0.292111 | 2.1184 | 64.3594 |
| candidate | 4580.030 | 136.530 | 264.625 | 0.291691 | 2.2775 | 63.9076 |

Both1088/1088 completion and full WER/SIM gates pass; both servers drain cleanly. Throughput changes only+0.0698%, makespan22.488465→22.755093s (+1.1856%), generated frames102926→104219 (+1.2562%). Zero waveforms have identical hashes;37 requests have equal frame counts. This is not a repeated/fixed-output speedup result. The option remains default off.

Actual candidate decisions: eligible159, capacity193, admission1216, audio_boundary142. Selected pairs are9.30% of1710 ordinary launches across64 warmup+1088 measured requests. Admission combines waiting and pending; the split is unknown. See pipeline-admission-audit.md for a source-based proposal, with no runtime edits.

Process-to-readiness was70.0338s control and52.0439s candidate, reported separately. Both use the same runtime/source and only the pipeline flag differs. Extra graph captures can change initial global sampling_step; later paired scheduling also changes row/step assignments. Full quality covers all outputs despite these differences.

One-second GPU process audits observed only each expected server and one quality process at a time; shorter unobserved interference cannot be excluded. Frozen source is clean and GPU6 released. Candidate is1.01547× frozen vLLM C64 baseline4510.2715fps, below target6765.4073fps.

Machine-readable evidence: pipeline-service-summary.json; service/quality declarations and commands; both raw benchmark directories and full quality JSONs; pipeline-service-process-audit.jsonl and pipeline-quality-process-audit.jsonl. Native prerequisite and retained failures: frame-pipeline-production-results.md.
