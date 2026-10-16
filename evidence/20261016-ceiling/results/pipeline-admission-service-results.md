# Deferred waiting admission: complete feasibility

Source95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b, clean detached source-pipeline-admission-95fcb0e. One C64 full1088 pair on GPU6; both FRAME_PIPELINE=1, only FRAME_PIPELINE_DEFERRED_WAITING0→1. Native graph/model unchanged from the3689-check gate;73 relevant CPU checks passed. Both options remain default off.

| Metric | Strict waiting control | Deferred waiting candidate |
|---|---:|---:|
| Throughput frames/s | 4481.280463 | 4583.673173 |
| TTFA p50 ms | 136.515141 | 146.277324 |
| TTFA p95 ms | 267.085166 | 274.848441 |
| RTF mean | 0.292684 | 0.289572 |
| Makespan s | 23.281069 | 22.209262 |
| Output frames | 104329.000000 | 101800.000000 |
| WER % | 1.992799 | 1.833710 |
| SIM | 64.041274 | 64.120952 |
| Process to readiness s | 64.033449 | 52.030748 |

Both1088 requests completed, all1088 WER and SIM scored with no filtering, both cleanly drained and passed WER≤2.9509336013% / SIM≥61.8815893412. Raw audio, full tail, command/launch metadata and all prior gate failures remain archived. GPU6 returned to4MiB/0%, frozen source clean. One-second process audit observed only each expected server or quality process, maximum one process onGPU6; shorter interference cannot be excluded.

Coverage is the clear result. Control decisions: eligible139, capacity370, pending0, waiting_runnable0, waiting_deferred1245, audio_boundary148. Candidate: eligible625, capacity218, pending0, waiting_runnable0, waiting_deferred0, audio_boundary403. Actual pair fraction among ordinary decode selections rises139/1902=7.3081%→625/1246=50.1605%. These lifetime counters include64 warmup plus1088 measured requests and exclude mixed/prefill branches. All1245 control admission rejections were waiting-only requests which the existing batching policy deferred. Candidate selected483 pairs with deferred waiting and142 with no waiting. The oldest queued age at selection averaged54.3558ms, maximum99.7474ms.

The single pair shows+2.2849% frames/s, with−2.4241% output frames and−4.6038% makespan; TTFA p50/p95 worsen9.762/7.763ms. Only2 PCM hashes and46 output frame counts match across arms. Both arms capture the same graphs, but global sampling_step was not independently measured/reset and later admission/row/step schedules differ; this is not a fixed-work or replicated speedup. Codec DP profiling/partitions also remain dynamic.

Candidate4583.673fps is only1.01627× the updated vLLM baseline4510.272, below6765.407 target. The coverage fix addresses real scheduling waste but does not justify a large throughput claim. Deadline crossings/new arrivals during a pair can wait one additional model step, not a fixed millisecond bound. Pending work, runnable admission, audio chunk horizons, maximum frames and native first-EOS append semantics remain unchanged.

Sources: [complete summary](pipeline-admission-service-summary.json), [declared service protocol](pipeline-admission-service-experiment.json), [quality protocol](pipeline-admission-quality-experiment.json), [CPU and launch verification](pipeline-admission-cpu-and-launch-verification.json), [CPU audit](pipeline-admission-audit.md).
