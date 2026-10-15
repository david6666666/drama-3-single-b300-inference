# Current concurrency optimization campaign

Status: **INCONCLUSIVE; the all-concurrency target has not been achieved.**
The latest user requirement supersedes the earlier 2× target: each concurrency
must achieve at least 1.5× vLLM-Omni `d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`.
WER may rise by at most one percentage point and SIM may fall by at most two
points, compared with the same concurrency. Both quality bounds are confirmed.
The [acceptance issue](https://github.com/ThinkFlowLab/OmniRSI/issues/10),
[frozen plan](../evidence/20261015-scale150/plan/plan.json) and
[validator](../evidence/20261015-scale150/validator/validate.py) require 1088
requests per repetition, three repetitions at each of seven concurrency levels,
complete quality coverage and matching source, hardware and workload controls.
No README success example will be added to OmniRSI before every gate passes.

## Complete baseline quality

All 21 vLLM performance groups now have complete WER/SIM scoring: 22,848 saved
waveforms, with no discarded outliers. WER aggregates the word-error counts
across all three repetitions; SIM is the mean of their per-sample scores.

| Concurrency | Required frames/s | vLLM WER % | vLLM SIM |
|---:|---:|---:|---:|
| 1 | 293.500 | 1.91465 | 64.05233 |
| 2 | 518.567 | 1.87278 | 64.05586 |
| 4 | 946.818 | 1.90349 | 64.07027 |
| 8 | 1652.346 | 1.73881 | 64.16508 |
| 16 | 2764.037 | 1.84487 | 63.92698 |
| 32 | 4296.985 | 1.85604 | 64.14505 |
| 64 | 6765.407 | 1.95093 | 63.88159 |

The validator independently rechecked the raw performance rows and all 21
quality groups. The new files supplement the nine historical quality groups;
they do not replace their outputs. See
[baseline completeness](../evidence/20261015-scale150/results/baseline-completeness.json)
and [quality aggregates](../evidence/20261015-scale150/results/vllm-quality-all-repetitions-summary.json).
No new candidate has complete evidence for all acceptance gates.

## Why scaling needs more than a faster decode kernel

An unchanged upstream service trace gives a median GPU span of 9.458 ms for
24 pure B64 decode graphs. The standalone heterogeneous B64 diagnostic is
8.589 ms, despite lower end-to-end throughput. These are diagnostic observations,
not a controlled serving speedup claim. Upstream kernel durations overlap and
their sum exceeds elapsed graph time. Comparing a mixed-batch average with a
full B64 operator would also misattribute the difference.

The standalone stage diagnostic spends substantial time outside decode:
9.18 s in prefill, 18.58 s in decode, and 4.36 s in codec, with mean decode
occupancy of 51.5 requests. Its synchronization changes execution, so these
totals explain hypotheses rather than predict an end-to-end speedup. Current
work targets padded prompt computation, admission occupancy and codec overlap,
while retaining numerical and state checks. A Rust rewrite is not yet supported
by evidence of a dominant CPU cost.

## Experiments and limitations

Each result below is a feasibility run or diagnostic, not a promoted three-run
performance claim. The unchanged formal candidate remains in
[the full comparison](full-comparison.md).

| Experiment | Observation | Decision |
|---|---|---|
| FlashInfer prefill MoE with corrected state isolation | C64 3527.70 frames/s; WER 1.817%, SIM 64.122; all 1088 outputs scored | Quality looks acceptable for this run; throughput remains far below 6765.41 |
| Six-request prefill bucket | State checks pass; C64 3466.27 frames/s | No substantial serving gain |
| Small MoE hot tiles | B8 pure decode 5.245 → 4.925 ms; layer tile errors within the diagnostic bound | Continue B2/B4 isolation before serving integration |
| Native small MoE versus FlashInfer at B8 | One real layer RRMS 0.03476 exceeds predeclared 0.03 | Do not change the dispatch default |
| MXFP8 expert re-encoding | Layer 0 RRMS 0.23045 versus native; independent implementation parity 0.01071 | Rejected by the predeclared 0.1 error limit; no service test |
| Packed variable-length attention primitive | CUDA Graph, changing lengths, empty sequences and causal GQA checks pass | Proceed only to isolated full-prefill validation |
| Complete packed prefill versus independent single requests | Worst state RRMS 0.322; unchanged padded control also reaches 0.394 | Both exceed 0.03; retain failed diagnostics and isolate each operation on identical inputs |

The original packed experiment was stopped at its numerical gate. A separately
declared layer-isolation experiment retains the 0.03 operator threshold and
exact state-ownership checks; it does not treat whole-network drift as a pass.
Neither graph parity nor layer parity replaces full audio quality. The frozen
acceptance plan and validator have not been relaxed.

Raw compact evidence and experiment plans are under
[20261015-scale150](../evidence/20261015-scale150).
Large profiler traces and audio stay in the recorded local artifact directories.
OmniRSI PR #9 has its conflicts resolved and a source-grounded concurrency
knowledge record, but no successful optimization claim.
