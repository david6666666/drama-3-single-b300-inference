The M128 global-buffer full-body candidate is numerically exact over the declared fixture, but fails the local continuation criterion. The three real layers improve only 1.304%, 0.569% and 2.871%; all were required to improve at least 5%. All 12 pressure scenarios avoid a >5% regression. The largest 7.519% result is a synthetic right-only routing distribution and does not substitute for real workloads. No model/service experiment or runtime integration follows this result.

Both measured graph calls include original native live metadata generation from identical precomputed full-top8 IDs/weights, FC1, native activation, FC2 and native finalizer. Each arm retains independent metadata, and intermediate comparisons use its actual mapping back to logical expanded route order. Router logits/topk selection and input quantization precede the archived inputs and are excluded. Baseline is vendor production tactic (8,5), PDL enabled; candidate is TS M128/four stages, PDL disabled. This is an implementation comparison, not a causal claim about MMA shape alone.

Three rounds use AB, BA, AB. Each arm warms up five graph replays, then two CUDA events bracket 100 complete calls. Device and wall spans are retained per round. There are no per-kernel timing events, profiler, cache flushes or clock changes; a two-second nvidia-smi ownership poll remains an observer. Medians below are over three device-span measurements. Startup/loading/compilation/hash checks are outside these spans; this small fixed-fixture test cannot estimate service readiness or throughput.

| Layer | Scenario | Vendor µs | M128 µs | Latency reduction |
|---:|---|---:|---:|---:|
| 0 | left_only | 81.855 | 79.460 | 2.926% |
| 0 | real | 120.787 | 119.212 | 1.304% |
| 0 | right_only | 82.708 | 79.957 | 3.326% |
| 0 | rows_reversed | 120.815 | 119.239 | 1.304% |
| 0 | worst_global288 | 149.401 | 144.901 | 3.012% |
| 19 | left_only | 64.448 | 59.657 | 7.433% |
| 19 | real | 99.553 | 98.987 | 0.569% |
| 19 | right_only | 69.656 | 64.419 | 7.519% |
| 19 | rows_reversed | 99.530 | 98.838 | 0.695% |
| 19 | worst_global288 | 149.936 | 144.918 | 3.347% |
| 39 | left_only | 63.445 | 59.848 | 5.669% |
| 39 | real | 80.682 | 78.366 | 2.871% |
| 39 | right_only | 66.464 | 63.475 | 4.497% |
| 39 | rows_reversed | 80.807 | 78.213 | 3.210% |
| 39 | worst_global288 | 149.467 | 144.581 | 3.269% |

The independent complete-body gate passed 1,253 checks over 15 scenarios (layers 0/19/39, real/reversed/left-only/right-only/worst288), including every FC1 and activation FP8 byte and FP32 scale, FC2/final BF16, guards, logical ownership, weights, inputs and RNG, eager and changed-metadata graphs. The timing run passed another 626 correctness/ownership checks. It adds explicit native/candidate metadata and workspace allocation-disjointness, map bounds and active-row-prefix checks. All original raw results and both earlier host/metadata harness failures remain unchanged.

M128 FC1 expanded passed 447 checks, including archived native zero/tiny inputs; selected-expert FC1 passed 542. FC2 expanded independently passed 306. Merged FC1/FC2 CUBINs match the corresponding component CUBINs exactly. Actual separate-kernel shared totals are 74,616 and 74,664 bytes, with CUBIN REG32 and REG31 and no stack/local spills. PTX register hints do not establish executed SETMAXREG or occupancy, and no occupancy-doubling claim is made.

Evidence: ../expert-m128-body-v3/result.json, ../expert-m128-body-v3/frozen-inputs.json, result.json, protocol.json, frozen-inputs.json, gpu.log, process-audit.jsonl and exit.json. GPU0 was idle before launch; task pid 668328 exited zero and post-process ownership was empty. Runtime and installed dependency sources were not edited. Source SHA 95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b remained clean.
