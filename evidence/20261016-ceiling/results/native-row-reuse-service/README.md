This is a CPU-prepared service feasibility controller, not a completed experiment.
The production source is pinned to `345c6a18f200853489a9ac8a86bbe0682ebe484d`.
Only `DRAMA3_MIXED_ROW_REUSE` changes from 0 to 1. The fixed controls, workload,
quality limits, measurement scope and stop rules are in `protocol-draft.json`.

The expanded production exact gate must pass before a launchable protocol can be
created. A failed or missing gate is rejected. Root's conditional authorization
also requires GPU6 to be released by the validation task. Do not bypass either
condition. Current draft preparation and its 30 CPU checks used no GPU queries or
Torch/CUDA import. The parser check uses prior complete 1088-request evidence and
does not constitute row-reuse correctness or performance evidence.

After the exact gate and independent review are complete:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 prepare_protocol.py --freeze
# Report the returned protocol/manifest hashes before execution.
PYTHONDONTWRITEBYTECODE=1 python3 run_experiment.py pair
PYTHONDONTWRITEBYTECODE=1 python3 run_experiment.py quality
```

Both phases refuse existing results. They preserve failures, source hashes,
commands, process ownership samples and cleanup reports. The quality phase scores
all 1088 outputs in each arm even if the valid service pair missed its throughput
gate. It refuses a failed or incomplete service run. No additional rounds are
launched automatically.

The candidate prebuilds five extra graphs, so initial sampling trajectories and
output lengths can differ. The summary reports full-run frames/s, TTFA p50/p95,
mean RTF, generated frame totals, readiness, sampled process memory, client intake
and drain, and lifetime row-reuse coverage. The mapping CPU counter covers only
planning/snapshots; validation and upload remain part of service wall time. No
drain or quality outlier is excluded. One feasible AB pair requires at least 1%
higher throughput and both original WER/SIM bounds; it is not a replicated win or
proof of the unchanged all-concurrency target. The C64 threshold remains
6765.407263214912 frames/s.
