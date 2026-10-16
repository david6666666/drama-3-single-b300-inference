# Mixed scheduling investigation: INCONCLUSIVE

The [frozen acceptance campaign](../20261015-scale150/plan/plan.json) and its
[validator](../20261015-scale150/validator/validate.py) are unchanged. The goal is
not met. These artifacts contain native-computation diagnostics and individual
full-corpus feasibility pairs; they are not the required three-repetition,
seven-concurrency acceptance matrix.

- `results/service-feasibility-summary.json`: mixed-backbone C64 control and
  candidate on GPU6, including complete 1088-output quality per arm. See raw
  `mixed-control-gpu6/` and `mixed-candidate-gpu6/` and both quality files.
- `results/mixed-layers*.json`: identical-input layer, graph and ownership
  checks. The expanded timing was contended and is unqualified. The failed
  duplicate-argument launch is retained as `mixed-layers-expanded.log.gz`.
- `results/vllm-continuous-schedule.json`: scheduler scopes and client phase
  analysis from a 256-request upstream diagnostic; graph fragments are not steps.
- `results/mixed-timeline-phase-analysis.json`: standalone 384-request stage
  spans, with admission and drain retained. These are not GPU utilization.
- `results/ncu/`: six actual-input B64 native expert GEMM counter samples,
  command, input/source provenance, raw metrics and limitations. The large raw
  Nsight report remains at its local artifact path with a recorded SHA-256.
- `results/mixed-span.json`: 339 exact output/state checks for shorter static
  prompt bounds at unchanged packed row capacity. `span-*-gpu0/` contains the
  separate full-corpus GPU0 feasibility pair, with quality scored independently.
- `results/limitations.json`: deviations, failed attempts and incomplete gates.

`results/files.json` records the uncompressed SHA-256 and byte count of every
selected file. JSONL/log files are gzip-compressed deterministically. Audio,
large traces, environments and source worktrees are excluded; raw local inputs
and paths remain in launch metadata. Never select a favorable subset of output
samples from the complete quality files.

Experimental flags default off. Mixed scheduling requires
`DRAMA3_MIXED_BACKBONE=1`, `DRAMA3_PACKED_PREFILL=1`, native FlashInfer prefill
MoE, and the existing eight-request prefill limit. The optional
`DRAMA3_MIXED_SPAN256=1` captures a shorter bound and retains a 512-bound fallback.
The source SHA and complete environment for each run are in its `launch.json`.

Preparation and process-to-readiness are reported separately from measured
requests. No caches were flushed, GPU clocks changed, experts pruned or extra
lossy quantization introduced. New success examples remain withheld until the
frozen campaign actually passes.


The later asynchronous shared/private-capture pair is also complete, with all
1088 outputs per arm: shared WER 6.163% fails; private WER 1.591% and SIM 64.191
pass the single-group bounds. Private throughput is 4360.917 frames/s, below
both baseline and target. `codec-dynamic/` retains the fixed-schedule failure
and exact private-capture controls; `codec-*-gpu0/` and both quality files retain
the full service pair. New async startup requires dedicated capture to prevent
the reproduced unsafe configuration. No formal acceptance verdict is promoted.
