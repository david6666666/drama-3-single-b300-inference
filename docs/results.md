# Measured streaming performance

Median of three complete runs. TTFA is in milliseconds; throughput is acoustic frames/s.

| Runtime | C | Frames/s | TTFA p50 | TTFA p95 | Mean RTF | Ratio to new vLLM |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 1 | 194.97 | 45.65 | 48.20 | 0.1112 | — |
| vllm | 2 | 342.46 | 54.52 | 61.06 | 0.1262 | — |
| vllm | 4 | 617.25 | 59.24 | 67.54 | 0.1390 | — |
| vllm | 8 | 1070.07 | 66.06 | 79.13 | 0.1589 | — |
| vllm | 16 | 1751.91 | 78.21 | 119.88 | 0.1883 | — |
| vllm | 32 | 2457.31 | 98.04 | 177.68 | 0.2361 | — |
| vllm | 64 | 3573.67 | 170.23 | 276.50 | 0.2849 | — |

This table alone does not establish acceptance. Complete quality coverage, matched
workloads and the recorded experiment controls must also pass. Missing evidence is
INCONCLUSIVE, never PASS.
