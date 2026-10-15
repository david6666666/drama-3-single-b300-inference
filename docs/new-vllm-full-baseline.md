# Measured streaming performance

Full Seed-TTS English corpus, 1088 requests per run, seven concurrency points,
three repetitions each: 22,848/22,848 successful measured requests. vLLM-Omni
is pinned at `d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`, including merged
PR #8933. One B300, GPU 0; identical pre-encoded reference codes and sampling
controls are used for the subsequent candidate. Per-C warmup is retained;
the redundant corpus-wide reference warmup is skipped for both runtimes.

All commands, deployment settings, readiness timing and raw records are in
[the baseline evidence](../evidence/20261015-b300/vllm-8933-codes1088).

Median of three complete runs. TTFA is in milliseconds; throughput is acoustic frames/s.

| Runtime | C | Frames/s | TTFA p50 | TTFA p95 | Mean RTF | Ratio to new vLLM |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 1 | 195.67 | 46.05 | 48.63 | 0.1107 | — |
| vllm | 2 | 345.71 | 54.50 | 60.45 | 0.1253 | — |
| vllm | 4 | 631.21 | 58.94 | 66.95 | 0.1373 | — |
| vllm | 8 | 1101.56 | 65.38 | 77.85 | 0.1571 | — |
| vllm | 16 | 1842.69 | 77.16 | 90.81 | 0.1863 | — |
| vllm | 32 | 2864.66 | 94.05 | 111.97 | 0.2354 | — |
| vllm | 64 | 4510.27 | 152.48 | 257.36 | 0.2859 | — |

This table alone does not establish acceptance. Complete quality coverage, matched
workloads and the recorded experiment controls must also pass. Missing evidence is
INCONCLUSIVE, never PASS.

Complete first-repetition quality: C1 WER 1.917%, SIM 64.273; C64 WER 1.842%,
SIM 63.871. Every quality point covers all 1088 requests without filtering.
The 2× throughput targets are 391.33, 691.42, 1262.42, 2203.13, 3685.38,
5729.31 and 9020.54 frames/s for C1, 2, 4, 8, 16, 32 and 64 respectively.
