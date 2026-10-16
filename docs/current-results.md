# Results at the end of this optimization phase

Snapshot: **2026-10-16**. Optimization paused at user request. The native model
constraint and target remain unchanged for any future work. The target has not
been met; the available evidence does not support an all-concurrency PASS.

The README contains the architecture diagram, optimization explanations and
these tables. This file records the comparison boundaries and source artifacts.
No new GPU run was started to produce this documentation.

## Measurement contract

Throughput = total returned PCM samples / 2048 / complete measured batch time.
TTFA = dispatch to first PCM bytes. Per-request RTF = dispatch-to-last-audio time
/ returned audio duration; each run reports its mean. Three-run rows aggregate
each metric independently by its median, including run-level TTFA percentiles.
Initial fill, final drain and 500-frame-capped outputs remain included. Raw PCM
does not expose EOS finish reasons. Sampling is temperature 1, top-p 0.9,
top-k 30, maximum 500 acoustic frames, mono 44100 Hz PCM.

No failed request or quality outlier is removed. The documentation audit
recomputed all four metrics for 63 archived runs and 49,728 successful requests
from raw timing/audio-length records, retaining each run's recorded makespan.
It verifies the tables, not comparability or full quality acceptance.
See [machine-readable values, pins, hashes and audit](current-results.json).

## Last complete repeated comparison

Standalone source `dcd20d34f0bc3d5511b149f495798317c31edc34`;
vLLM source `d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`.
Each has 1088 requests × seven concurrency levels × three repetitions. Same GPU0,
CPU affinity, OMP=8, model snapshot, encoded reference integers and client.
Standalone capacity 64, upstream 128; BF16 dense/GDN and original FP8 experts,
standalone codec BF16 versus upstream FP16. Warmup is max(8,C); reference encoding
is prepared separately; fresh TCP per request, no retries.

| Framework | C | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---|---:|---:|---:|---:|---:|
| drama-3-single-b300-inference | 1 | 300.81 | 33.32 | 34.70 | 0.0722 |
| vllm-omni | 1 | 195.67 | 46.05 | 48.63 | 0.1107 |
| drama-3-single-b300-inference | 2 | 466.46 | 48.04 | 63.43 | 0.0932 |
| vllm-omni | 2 | 345.71 | 54.50 | 60.45 | 0.1253 |
| drama-3-single-b300-inference | 4 | 753.61 | 44.19 | 62.25 | 0.1149 |
| vllm-omni | 4 | 631.21 | 58.94 | 66.95 | 0.1373 |
| drama-3-single-b300-inference | 8 | 1091.64 | 53.36 | 77.56 | 0.1585 |
| vllm-omni | 8 | 1101.56 | 65.38 | 77.85 | 0.1571 |
| drama-3-single-b300-inference | 16 | 1591.86 | 75.76 | 103.45 | 0.2164 |
| vllm-omni | 16 | 1842.69 | 77.16 | 90.81 | 0.1863 |
| drama-3-single-b300-inference | 32 | 2167.10 | 99.29 | 146.30 | 0.3150 |
| vllm-omni | 32 | 2864.66 | 94.05 | 111.97 | 0.2354 |
| drama-3-single-b300-inference | 64 | 3451.15 | 156.33 | 414.23 | 0.3872 |
| vllm-omni | 64 | 4510.27 | 152.48 | 257.36 | 0.2859 |

Sources: [standalone](../evidence/20261015-b300/candidate-prefill192-codes1088-formal),
[vLLM](../evidence/20261015-b300/vllm-8933-codes1088),
[controls and figure](full-comparison.md), [commands](reproduce-full-benchmark.md).
The historical pre-fix candidate is excluded because of its state-isolation bug.

## Historical SGLang measurements

Source `e9e016b83e1a5d92b4232f90427a44afac8d84b6`, capacity 64, GPU0, OMP=8.
192 raw-audio-reference requests × seven concurrency levels × three repetitions;
reference warmup enabled. The corpus, reference handling and connection policy
differ from the 1088 coded-reference runs. These rows are useful historical
measurements but cannot establish a three-way ranking or speedup ratio.
The pinned SGLang public adapter was not used with the later coded corpus;
no matched 1088-request SGLang matrix or complete all-repetition quality gate
exists in this campaign. No missing measurement is replaced with an estimate.

| Framework | C | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---|---:|---:|---:|---:|---:|
| sglang-omni | 1 | 168.02 | 78.98 | 82.14 | 0.1296 |
| sglang-omni | 2 | 289.90 | 88.42 | 113.64 | 0.1501 |
| sglang-omni | 4 | 478.60 | 98.54 | 140.30 | 0.1801 |
| sglang-omni | 8 | 705.36 | 136.18 | 183.08 | 0.2425 |
| sglang-omni | 16 | 953.27 | 191.04 | 251.58 | 0.3456 |
| sglang-omni | 32 | 1309.37 | 270.67 | 394.37 | 0.4880 |
| sglang-omni | 64 | 1773.49 | 439.54 | 632.33 | 0.6816 |

Source: [launch, manifest, command and raw timing records](../evidence/20261014-b300/sglang-cap64).

## Later standalone C64 measurements

| C64 configuration | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) | WER (%) | SIM |
|---|---:|---:|---:|---:|---:|---:|
| Best single observation, `e7cda1e` | 4726.77 | 145.65 | 273.68 | 0.2881 | 1.599 | 64.413 |
| Latest control, reuse off, `345c6a1` | 4674.81 | 148.56 | 274.70 | 0.2909 | 1.465 | 64.188 |
| Latest candidate, reuse on, `345c6a1` | 4534.13 | 151.27 | 273.35 | 0.2906 | 1.683 | 64.202 |

All three rows are single observations on GPU6, with complete 1088-output WER/SIM.
Only the last two rows are a same-source feasibility pair. The earlier best
observation is not a replicated winner and does not replace the repeated C64 row.
The paired candidate/controller ratio is 0.969905981; the +1% continuation gate
fails. Both pass WER≤2.950934% and SIM≥61.881589. No repeated service trial follows.
Returned output lengths differ and the full tail remains included.

Sources: [best single observation](../evidence/20261016-ceiling/results/codec-cache-service-summary.json),
[latest pair](../evidence/20261016-ceiling/results/native-row-reuse-service/summary.json),
[latest pair interpretation](../evidence/20261016-ceiling/results/native-row-reuse-service/summary.md).

## What remains unestablished

- Latest standalone C1/2/4/8/16/32 performance and a repeated latest C64 winner.
- A common-workload three-framework matrix with complete quality coverage.
- The 1.5× target at all seven concurrency levels, including C64 6765.407263 frames/s.
- An end-to-end theoretical ceiling: kernel-only arithmetic and sampled bandwidth
  are not complete service limits.

The frozen requirement is all 1088 outputs × three repetitions × seven concurrency
levels, with WER at most vLLM +1 percentage point and SIM at least vLLM −2 at each
concurrency. All 21 baseline quality groups are available; latest standalone
coverage is limited to the documented feasibility groups. See
[baseline quality](scale150-progress.md#complete-baseline-quality) and
[historical candidate quality coverage](full-quality.md).
