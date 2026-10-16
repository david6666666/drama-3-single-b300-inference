# Cached codec Linear weights: complete service feasibility

Sourcee7cda1eba083603b88e1f2e89688e1ff2ced7fc2, clean source-codec-cache worktree. GPU6, C64, one complete1088-request pair; both FRAME_PIPELINE1, DEFERRED_WAITING1, reserved-router1/five native flags, private async codec/thread, DP, coalesce0/SMS0/defer0. Actual launch metadata differs only in DRAMA3_CODEC_CACHED_LINEAR0→1.

The production gate passed before this run:46 captured shapes exact PCM/all mutable state; normal_sync/cached_sync/cached_poll/cached_thread each414 dynamic trace entries and all80 PCM/full-model+codec states exact, distinct BLAS workspaces. The candidate caches native FP32→BF16 Linear conversions; original loaded codec/reference weights and norm precision remain intact. A reversed conversion-direction typo in the original protocol is explicitly corrected in codec-cache-protocol-correction.json; executed source/flags were correct and the original driver/hash was preserved.

| Metric | Control | Cached Linear |
|---|---:|---:|
| Throughput frames/s | 4726.771495 | 4585.126356 |
| TTFA p50 ms | 145.645105 | 142.863836 |
| TTFA p95 ms | 273.681829 | 276.595375 |
| RTF mean | 0.288130 | 0.288258 |
| Makespan s | 21.479566 | 22.707117 |
| Output frames | 101529.000000 | 104115.000000 |
| WER % | 1.599263 | 2.017918 |
| SIM | 64.412759 | 63.860895 |
| Process to readiness s | 50.031823 | 68.039176 |

Both1088/1088 requests succeeded and drained; both complete unfiltered1088 WER/SIM evaluations passed WER≤2.9509336013% / SIM≥61.8815893412. No samples or tails were removed. Candidate WER rises0.4187pp and SIM falls0.5519 versus same-source control, within the user limits and also within same-source1pp/2 thresholds.

This is a negative service feasibility: throughput−2.9967%, makespan+5.7150%, output frames+2.5471%, RTF nearly unchanged. Only51 PCM hashes/101 frame counts match. Codec DP profiles its own costs each startup, and scheduling can change sampler row/step histories and output lengths; a single variable native optimization does not imply fixed output. This one pair neither proves a repeatable regression nor supports promoting the local codec speedup as a service gain. Keep the option default off; no6765 target claim.

Actual ordinary-decode decisions, including64 warmup+1088 measured requests: control eligible603/capacity142/audio_boundary456; candidate599/301/487. Neither saw pending or runnable/deferred-waiting rejections because both deferred policies are enabled. Eligible with deferred waiting is467 in each arm. These lifetime counters exclude mixed/prefill paths.

Observed process memory (MiB):
- Control: {"startup_samples": 40, "service_samples": 23, "startup_peak": 105760, "service_peak": 105788, "service_median": 105788, "limitation": "One-second nvidia-smi process sampling. Readiness boundary approximated from prelaunch wall clock plus recorded process-to-health time; cannot resolve subsecond clone transients or allocator peak."}
- Candidate: {"startup_samples": 57, "service_samples": 23, "startup_peak": 107030, "service_peak": 107058, "service_median": 107058, "limitation": "One-second nvidia-smi process sampling. Readiness boundary approximated from prelaunch wall clock plus recorded process-to-health time; cannot resolve subsecond clone transients or allocator peak."}

Observed total process peak increases105788→107058MiB (+1270MiB), exceeding the approximately208MiB retained native weight cache because process memory includes allocator/capture storage. One-second polling cannot resolve transient deep-copy peaks; no exact allocator peak is claimed. Readiness costs above include startup/capture and are reported separately from request throughput; no download/copy/shared-cache flush or clock changes.

GPU6 process audit observed only expected server PIDs482793/485471 and quality PIDs488769/491390, maximum one onGPU6; polling cannot exclude subsecond interference. GPU6 released, frozen worktree clean. Native gate success and negative service evidence are both retained.

Evidence: [complete summary](codec-cache-service-summary.json), [service declaration](codec-cache-service-experiment.json), [quality declaration](codec-cache-quality-experiment.json), [native dynamic gate](codec-cache-dynamic/result.json), [protocol wording correction](codec-cache-protocol-correction.json).
