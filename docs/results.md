# Measured streaming performance

**Correctness update:** the historical candidate below has a confirmed GDN
state-isolation defect for partially filled prefill batches. A dummy row's
suffix slot could overwrite another live request. Its concurrent results
cannot support promotion, even where sampled WER met provisional bounds.
The baseline is unaffected. Commit `a6e660c` reserves separate primary/suffix
padding slots; corrected full-corpus runs are in progress. Historical rows and
raw results remain here for auditability.

Median of three complete runs. TTFA is in milliseconds; throughput is acoustic frames/s.

| Runtime | C | Frames/s | TTFA p50 | TTFA p95 | Mean RTF | Ratio to new vLLM |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 1 | 194.97 | 45.65 | 48.20 | 0.1112 | — |
| candidate | 1 | 289.88 | 36.58 | 37.34 | 0.0756 | 1.487× |
| vllm | 2 | 342.46 | 54.52 | 61.06 | 0.1262 | — |
| candidate | 2 | 428.10 | 52.31 | 66.87 | 0.1016 | 1.250× |
| vllm | 4 | 617.25 | 59.24 | 67.54 | 0.1390 | — |
| candidate | 4 | 685.32 | 48.94 | 69.10 | 0.1250 | 1.110× |
| vllm | 8 | 1070.07 | 66.06 | 79.13 | 0.1589 | — |
| candidate | 8 | 976.03 | 59.91 | 98.36 | 0.1760 | 0.912× |
| vllm | 16 | 1751.91 | 78.21 | 119.88 | 0.1883 | — |
| candidate | 16 | 1429.82 | 83.19 | 165.38 | 0.2345 | 0.816× |
| vllm | 32 | 2457.31 | 98.04 | 177.68 | 0.2361 | — |
| candidate | 32 | 1885.99 | 109.70 | 335.21 | 0.3279 | 0.768× |
| vllm | 64 | 3573.67 | 170.23 | 276.50 | 0.2849 | — |
| candidate | 64 | 2279.34 | 174.25 | 641.66 | 0.4892 | 0.638× |

This table alone does not establish acceptance. Complete quality coverage, matched
workloads and the recorded experiment controls must also pass. Missing evidence is
INCONCLUSIVE, never PASS.

Throughput-only 2× gate: **FAIL**. Quality is a separate required gate.

## Quality and controls

Same 192 English Seed-TTS requests with pre-encoded references; each throughput
row summarizes three runs on GPU 0. Candidate source is `4f97185`, baseline
source is `d6ed4f4` including PR #8933. Both completed 4032/4032 measured requests.
Candidate dense weights/activations remain BF16 and recurrent state FP32;
the upstream deployment uses BF16 recurrent state. No FP8-dense experiment is
included in this table.

| Runtime | C | WER (%) | Speaker SIM | Coverage |
|---|---:|---:|---:|---:|
| vLLM | 1 | 2.227 | 62.411 | 192/192 |
| Candidate | 1 | 2.326 | 62.055 | 192/192 |
| vLLM | 64 | 1.880 | 62.738 | 192/192 |
| Candidate | 64 | 2.672 | 61.855 | 192/192 |

Quality uses complete first-repetition audio at C1 and C64, with no high-WER
filtering. These points are within the provisional tolerances of baseline WER
+1 percentage point and SIM −2 points. These are working assumptions, not
user-confirmed quality requirements, and do not override the failed throughput
gate. The 192-row subset does not establish quality over the full 1088-row corpus.
Raw per-request timings, failures, hashes and launch metadata are under
`evidence/20261015-b300/`.
