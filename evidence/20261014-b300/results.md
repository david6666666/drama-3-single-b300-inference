# Measured streaming performance

Median of three complete runs. TTFA is in milliseconds; throughput is acoustic frames/s.

| Runtime | C | Frames/s | TTFA p50 | TTFA p95 | Mean RTF | Ratio to better baseline |
|---|---:|---:|---:|---:|---:|---:|
| sglang | 1 | 168.02 | 78.98 | 82.14 | 0.1296 | — |
| sglang | 2 | 289.90 | 88.42 | 113.64 | 0.1501 | — |
| sglang | 4 | 478.60 | 98.54 | 140.30 | 0.1801 | — |
| sglang | 8 | 705.36 | 136.18 | 183.08 | 0.2425 | — |
| sglang | 16 | 953.27 | 191.04 | 251.58 | 0.3456 | — |
| sglang | 32 | 1309.37 | 270.67 | 394.37 | 0.4880 | — |
| sglang | 64 | 1773.49 | 439.54 | 632.33 | 0.6816 | — |

This table alone does not establish acceptance. Complete quality coverage, matched
workloads and the recorded experiment controls must also pass. Missing evidence is
INCONCLUSIVE, never PASS.
