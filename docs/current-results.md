# Matched three-framework B300 comparison

Sources were pinned on **2026-10-16**. This report replaces the README's earlier
mixture of old full-matrix measurements and newer standalone C64 observations.
All three frameworks use the same 1088 pre-encoded-reference requests per planned
run. The vLLM C64 third repetition stalled, and SGLang C2 stopped after an HTTP
500 in its second repetition. Their three-run audits remain incomplete. At the
user's request, the performance display now reuses the two complete vLLM C64
runs and the one complete SGLang C2 run, with their actual repeat counts marked.
No new experiments were run for this update. Source versions and inputs are unchanged.
The SGLang adapter was corrected to pass its supported `references[].vq_codes`
field; this is a benchmark-client correction, not a change to the model.

Optimization remains paused. These are descriptive measurements of the existing
implementations, not a claim that the historical all-concurrency 1.5× target was
achieved. [Historical results](historical-results-20261016.md) retain their original
versions and measurement boundaries.

## Versions and controls

| Framework | Source | Base runtime | Torch | FlashInfer |
|---|---|---|---|---|
| drama-3-single-b300-inference | [33f6d72](https://github.com/david6666666/drama-3-single-b300-inference/commit/33f6d7237196520be9f457d86df161264cbb087e) | dedicated Python engine | 2.13.0 | 0.7.0.post1 |
| vllm-omni | [aa9c9a2](https://github.com/vllm-project/vllm-omni/commit/aa9c9a2d2ce3ff8640f1b4ab8937a6194b0ead4d) | 0.31.0 | 2.13.0+cu130 | 0.7.0.post1 |
| sglang-omni | [2c23b68](https://github.com/sgl-project/sglang-omni/commit/2c23b68f4f763c14cfb845eadb4883a5e791b701) | 0.5.21 | 2.13.0 | 0.6.18 |

The upstream source commits were the fetched main heads at preparation time;
the standalone runtime is frozen at the client-fix commit. Existing final-path
environments were reused without dependency upgrades. The installed vLLM-Omni
distribution metadata still carries an older editable-build version; CPU imports
and live launch records bind its actual modules to the new pinned worktree.
[Preparation and exact dependency versions](../evidence/20261016-readme-comparison/files/preparation.json).
SGLang's native `sm_103` dispatch uses PyTorch codebook sampling and disables
fused tower layers; its installed MoE kernels report generic configurations
because B300-specific tuning files are absent. These choices were recorded,
not overridden. The comparison does not establish each framework's performance
ceiling over all possible dependency versions and tuning configurations.

- Same physical B300: GPU6, UUID `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`; OMP=8 and CPU affinity 0–239.
- Same Drama 3 checkpoint `e0a9218aa48888270f06ef1ed9ea67622c0e17d2` and Fish Speech `541566932abbaebcb0a40e562a1f846738383f38`.
- Same ordered English Seed-TTS corpus, SHA-256 `08612ee05ab0dd8a762109978b5f230e5b0311a1dfde87f03a4572434dff2dac`; all reference-code integers were checked across the three public adapters.
- Native FP8 experts and selected top-8 routes retained. Standalone uses [this profile](../configs/benchmark-native-b300.env), 64 slots, BF16 GDN/codec and asynchronous stateful DAC. vLLM uses its native two-stage `fish_omni.yaml`, private MPS, capacity 128 and FP16 stateful codec. SGLang uses capacity 64 and its native FP16 window codec. These are framework implementations, not identical operator configurations.
- Temperature 1, top-p 0.9, top-k 30, maximum 500 frames. Fresh TCP for every request, no retries. Before each concurrency: max(8,C) warmup requests, outside measured time. Each framework initially planned one server for 21 groups. vLLM stopped in its last group. SGLang stopped in C2; only its unattempted C4–C64 groups used a new server, with identical parameters and separate warmups. That restart resets serving history and can affect RNG state; this is not one continuous SGLang matrix session.
- A separate C64 feasibility run and full quality pass preceded the matrix. Feasibility values are excluded from the table. Preparation, process-to-readiness, warmup, measurement and scoring remain separate in the logs.

[Fixed protocol](../evidence/20261016-readme-comparison/files/protocol.json) · [Recovery matrix protocol](../evidence/20261016-readme-comparison/files/recovery-attempt-01/matrix-protocol.json) · [Post-vLLM continuation](../evidence/20261016-readme-comparison/files/interrupted-matrix-report-01/protocol.json) · [Remaining SGLang protocol](../evidence/20261016-readme-comparison/files/remaining-sglang-report-01/protocol.json) · [Machine-readable summary](current-results.json)

The original matrix stopped after seven complete groups when a whole-machine
GPU-monitor query exceeded its 10-second timeout. Those groups and the interrupted
output remain archived and are **excluded** from this table. The recovery attempt
restarted all 21 standalone groups and then started vLLM. SGLang followed under
the continuation described below. The recovery attempt's only controller change was UUID-scoped monitoring with a 30-second timeout; any timeout
still stopped the run. Model sources, dependencies and benchmark settings were
unchanged. [Stop audit](../evidence/20261016-readme-comparison/files/matrix-stop-audit.json).

## Interrupted vLLM C64 repetition

The final vLLM group, `c64-r2`, did not finish. Its last normal service log entries
were recorded at 15:28:38 UTC. At approximately 15:33:39, server cancellation
records appeared after the frozen
300-second client deadline, followed by stage-resource-release timeouts. The
controller was then interrupted and cleaned only its own processes; GPU6 was
empty afterwards. The root cause has not been established, and this group was
not retried. [Failure audit](../evidence/20261016-readme-comparison/files/recovery-attempt-01/vllm-stop-audit.json).

The first 20 vLLM groups contain 21,760 fully persisted, validated requests.
The client buffers a group until it finishes, so no complete `c64-r2` request/WAV
group was saved. Its exact success/failure totals are unavailable. Server logs
contain 72 cancellation lines for 64 distinct IDs; these are observed
cancellations, not a complete failed-request count.

Only the 20 complete vLLM groups were scored. A separately frozen continuation
then ran SGLang with unchanged source, model, client, hardware and test parameters.
All original failures remain archived.

## SGLang C2 failure and continuation

SGLang completed C1 three times and C2 once. In `c2-r1`, request
`c2-r1-00351` returned HTTP 500: `Drama 3 generated no audio frames`.
The group persisted all 1088 request records, with **1087 successes and one failure**.
The client stopped after that invalid group; `c2-r2` was not run. The controller
cleaned its own processes, leaving GPU6 empty. This error identifies the observed
failure, not its underlying model/sampling cause.
[Failure audit](../evidence/20261016-readme-comparison/files/interrupted-matrix-report-01/sglang-stop-audit.json).

All saved audio was scored, including the 1087 outputs from the invalid group.
The failed request retains its error and is not treated as a scored output. Its partial
scores are diagnostics, not a quality score for all 1088 requested samples.
Neither the failed request nor any completed C1/C2 group was retried.
A separately frozen continuation measured only the previously unattempted
C4, C8, C16, C32 and C64 groups, three times each, then scored all their outputs.
Sources, inputs, capacities, sampling and warmup policy remained unchanged.
The new SGLang process is an explicit session boundary; its serving/RNG history
differs from the interrupted process, and each row records its source session.

These individual observations preserve the evidence for the incomplete three-run
rows. The three complete groups below now supply the marked performance display
rows; they are still not formal three-run estimates. The failed SGLang group remains
**diagnostic only** and is excluded from the display aggregates and SGLang ratio
baseline. Its throughput uses returned audio and the whole group makespan;
TTFA/RTF and quality cover only its 1087 successful outputs:

| Framework / run | Success / failed | frames/s | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) | WER (%) | SIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni / c64-r0 | 1088 / 0 | 4551.91 | 155.00 | 263.80 | 0.2860 | 2.144 | 64.197 |
| vllm-omni / c64-r1 | 1088 / 0 | 4729.74 | 156.08 | 282.87 | 0.2863 | 2.085 | 63.887 |
| sglang-omni / c2-r0 | 1088 / 0 | 295.39 | 87.57 | 112.38 | 0.1470 | 1.859 | 64.276 |
| sglang-omni / c2-r1 | 1087 / 1 | 294.96 | 87.79 | 111.61 | 0.1473 | 1.634 | 64.097 |

## Performance

| C | Framework | Throughput (frames/s) | Throughput / SGLang | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | drama-3-single-b300-inference | 309.17 | 1.82× | 32.83 | 44.68 | 0.0702 |
| 1 | vllm-omni | 195.89 | 1.15× | 47.29 | 49.76 | 0.1106 |
| 1 | sglang-omni | 169.80 | 1.00× | 79.38 | 82.82 | 0.1278 |
| 2 | drama-3-single-b300-inference | 479.42 | 1.62×‡ | 47.22 | 58.04 | 0.0908 |
| 2 | vllm-omni | 346.29 | 1.17×‡ | 54.25 | 60.56 | 0.1251 |
| 2 | sglang-omni | 295.39‡ | 1.00×‡ | 87.57 | 112.38 | 0.1470 |
| 4 | drama-3-single-b300-inference | 785.51 | 1.62× | 44.07 | 60.33 | 0.1104 |
| 4 | vllm-omni | 631.60 | 1.30× | 58.93 | 66.82 | 0.1371 |
| 4 | sglang-omni | 485.32 | 1.00× | 99.71 | 137.73 | 0.1788 |
| 8 | drama-3-single-b300-inference | 1153.25 | 1.61× | 53.87 | 79.31 | 0.1500 |
| 8 | vllm-omni | 1105.75 | 1.54× | 65.53 | 77.70 | 0.1564 |
| 8 | sglang-omni | 715.78 | 1.00× | 132.30 | 180.38 | 0.2421 |
| 16 | drama-3-single-b300-inference | 1857.42 | 1.85× | 74.94 | 100.39 | 0.1858 |
| 16 | vllm-omni | 1846.24 | 1.84× | 77.44 | 90.50 | 0.1861 |
| 16 | sglang-omni | 1002.32 | 1.00× | 184.97 | 248.58 | 0.3444 |
| 32 | drama-3-single-b300-inference | 2652.72 | 1.91× | 96.38 | 127.34 | 0.2551 |
| 32 | vllm-omni | 2878.10 | 2.07× | 95.60 | 113.42 | 0.2348 |
| 32 | sglang-omni | 1389.46 | 1.00× | 256.62 | 337.24 | 0.4916 |
| 64 | drama-3-single-b300-inference | 4662.02 | 2.47× | 147.71 | 275.61 | 0.2918 |
| 64 | vllm-omni | 4640.83† | 2.46×† | 155.54 | 273.34 | 0.2862 |
| 64 | sglang-omni | 1886.22 | 1.00× | 384.36 | 619.29 | 0.7058 |

Rows use independent medians of three complete 1088-request groups, except:

- † vLLM C64: the independent median of the two complete groups, `c64-r0` and `c64-r1` (for two values, their arithmetic mean). The timed-out third group remains recorded and contributes no invented measurement.
- ‡ SGLang C2: the one complete group, `c2-r0`. Its throughput is also the denominator for all C2 ratios; the failed second group and unattempted third group remain recorded.

`Throughput / SGLang` = the row's unrounded throughput / the unrounded SGLang
throughput at the same concurrency. It is a ratio of these descriptive aggregates,
not a median of paired-run ratios. The [JSON report](current-results.json) keeps
these 21 display rows under `performance_table`, including repeat counts and
source labels; the original 19 formal rows, incomplete-row audit and quality
acceptance status remain unchanged. No new experiments were run.

Throughput = total returned PCM samples / 2048 / full group makespan.
TTFA p50/p95 = per-run percentiles of dispatch-to-first-PCM time, then medians
across the included complete runs. RTF = each request's dispatch-to-last-PCM time /
audio duration; the table aggregates those run means in the same way. Audio is mono PCM16 at
44.1 kHz. Initial fill, final drain and frame-capped outputs are included.
Raw PCM does not expose EOS reasons. Generated lengths and random-number
schedules differ; no equal-output or bitwise-equivalence claim is made.

The original three-run audit remains below (ranges are not confidence intervals):

| C | Standalone frames/s range | vLLM-Omni frames/s range | SGLang-Omni frames/s range |
|---:|---:|---:|---:|
| 1 | 308.78–309.18 | 195.49–195.95 | 168.33–170.19 |
| 2 | 478.88–479.48 | 346.17–346.53 | Incomplete (1 valid / 1 failed / 1 not run) |
| 4 | 785.15–786.62 | 630.38–632.38 | 485.23–490.09 |
| 8 | 1148.71–1157.37 | 1105.06–1107.12 | 715.33–727.74 |
| 16 | 1853.16–1869.74 | 1844.85–1848.58 | 987.95–1013.95 |
| 32 | 2646.19–2668.71 | 2816.77–2903.41 | 1348.88–1393.54 |
| 64 | 4482.08–4668.73 | Incomplete (2/3) | 1880.50–1894.69 |

All four metrics were independently recomputed from raw records. Every measured
WAV's PCM hash, sample count and format were verified for **65,280 requests in
60 complete groups**, plus **1087 saved outputs from the partial SGLang group**.
No successful output was removed. Of the 63 planned groups, 60 completed without
request errors, one SGLang group contains a recorded failure, one SGLang group
was not run, and the vLLM C64 final group is incomplete with unknown final counts.
Missing or failed requests are not counted as successful. The monitor queried only
the selected GPU UUID, with a one-second pause between queries and a 30-second
timeout. Actual poll timestamps are retained; this cannot
exclude interference between observations. Other GPUs kept their existing jobs.

## Audio quality

Every persisted output was scored: **66,367 / 66,367**, comprising 65,280 outputs
from 60 complete groups and 1087 from the partial SGLang group. This excludes
the one no-audio HTTP 500, the unattempted group and the unsaved vLLM group; their
missing audio is not assigned a fabricated quality score. Fixed Whisper large-v3 greedy
English ASR and the pinned SGLang WER/WavLM SIM tools were reused. WER is the
unfiltered corpus word-error rate; SIM is the mean across all successfully
generated and scored outputs in that group. The following numeric values are medians of three complete run scores; per-run scores
and ranges are preserved in [the JSON report](current-results.json).

| C | Standalone WER (%) / SIM | vLLM-Omni WER (%) / SIM | SGLang-Omni WER (%) / SIM |
|---:|---:|---:|---:|
| 1 | 1.817 / 63.903 | 1.917 / 64.249 | 2.093 / 63.883 |
| 2 | 1.901 / 63.977 | 2.026 / 64.135 | Incomplete (1 valid / 1 failed / 1 not run) |
| 4 | 2.060 / 63.891 | 1.675 / 64.100 | 1.993 / 64.172 |
| 8 | 2.077 / 63.994 | 1.901 / 63.960 | 1.825 / 64.149 |
| 16 | 1.691 / 64.110 | 1.817 / 64.278 | 1.926 / 63.756 |
| 32 | 2.077 / 63.999 | 2.060 / 64.084 | 2.127 / 63.854 |
| 64 | 1.591 / 64.308 | Incomplete (2/3) | 2.035 / 63.961 |

At C1–C32, every standalone repetition remains within vLLM +1 percentage point WER / −2 SIM. C64 has no complete three-run vLLM baseline, so no corresponding quality-bound verdict is reported.
Available bounds are reported against the new, same-concurrency vLLM measurements;
no score was used to remove a performance result or select a better repetition.
This does not retarget or declare PASS for the earlier optimization requirement.
Repeat labels do not imply matching random outputs.

## Reproduction and evidence

Use the README installation and serve commands. For the measured standalone
configuration, load the profile before starting the server:

```bash
set -a
. configs/benchmark-native-b300.env
set +a
```

The pinned `scripts/run_baseline.py` and `scripts/evaluate_audio.py` provide the
shared client and scorer. Exact executable commands, source bindings, capacities,
startup times, raw timing records, quality records and validation summaries are
in [the evidence archive](../evidence/20261016-readme-comparison/README.md). WAV audio and model weights stay
outside Git; each successful audio request record retains its returned PCM SHA-256.

The former unmatched SGLang 192-request measurements and older standalone
versions remain in [the historical archive](historical-results-20261016.md).
