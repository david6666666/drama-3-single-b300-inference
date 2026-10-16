"""Render documentation only from an audited comparison summary."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ART = Path(__file__).resolve().parent
NAMES = {'standalone': 'drama-3-single-b300-inference', 'vllm': 'vllm-omni', 'sglang': 'sglang-omni'}
REPOS = {'standalone': 'david6666666/drama-3-single-b300-inference', 'vllm': 'vllm-project/vllm-omni', 'sglang': 'sgl-project/sglang-omni'}
KEYS = ['throughput_frames_s', 'ttfa_p50_ms', 'ttfa_p95_ms', 'rtf_mean']
ORDER = ['standalone', 'vllm', 'sglang']
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]
EVIDENCE = '../evidence/20261016-readme-comparison'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    data = read(ART / 'remaining-sglang-report-01/matrix-summary.json')
    protocol = read(ART / 'protocol.json')
    assert data['status'] == 'COMPLETE_DESCRIPTIVE_COMPARISON_WITH_INCOMPLETE_ROWS'
    assert data['performance_groups'] == data['quality_groups'] == 60
    assert data['partial_performance_groups'] == data['partial_quality_groups'] == 1
    assert data['partial_scored_requests'] == 1087 and data['total_scored_requests'] == 66367
    assert data['complete_group_requests'] == 65280 and not data['complete_matrix']
    incomplete = {(r['framework'], r['concurrency']): r for r in data['incomplete_rows']}
    assert set(incomplete) == {('vllm', 64), ('sglang', 2)}
    missing_marks = {('vllm', 64): '—†', ('sglang', 2): '—‡'}
    lookup = {(r['framework'], r['concurrency']): r for r in data['rows']}
    assert len(lookup) == 19 and not (set(lookup) & set(incomplete))
    rows = [lookup[(b, c)] for c in CONCURRENCY for b in ORDER if (b, c) in lookup]
    pins = {b: protocol['source_shas'][b] for b in ORDER}
    pin_links = {b: f"[{pins[b][:7]}](https://github.com/{REPOS[b]}/commit/{pins[b]})" for b in ORDER}
    table = ['| C | Framework | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |',
             '|---:|---|---:|---:|---:|---:|']
    for c in CONCURRENCY:
        for backend in ORDER:
            row = lookup.get((backend, c))
            values = ([f"{row['metrics'][k]['median']:.{4 if k == 'rtf_mean' else 2}f}" for k in KEYS]
                      if row else [missing_marks[(backend, c)]] * 4)
            table.append(f"| {c} | {NAMES[backend]} | " + ' | '.join(values) + ' |')
    table = '\n'.join(table)
    comparable = [r for r in rows if r['framework'] == 'standalone' and r['quality_vs_vllm']['available']]
    assert len(comparable) == 6
    bounds = all(r['quality_vs_vllm']['all_repeats_within_reporting_bounds'] for r in comparable)
    matrix_quality_note = ('At C1–C32, every standalone repetition remains within vLLM +1 percentage point WER / −2 SIM.'
                           if bounds else 'At C1–C32, some standalone quality results exceed vLLM +1 percentage point WER / −2 SIM; see the per-run report.')
    matrix_quality_note += ' C64 has no complete three-run vLLM baseline, so no corresponding quality-bound verdict is reported.'
    diagnostic_table = ['| Framework / run | Success / failed | frames/s | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) | WER (%) | SIM |',
                        '|---|---:|---:|---:|---:|---:|---:|---:|']
    assert len(data['diagnostic_groups']) == 4
    for group in data['diagnostic_groups']:
        m = group['performance']; q = group['quality']
        values = [f"{m[k]:.{4 if k == 'rtf_mean' else 2}f}" for k in KEYS]
        diagnostic_table.append(f"| {NAMES[group['framework']]} / {group['label']} | {group['successful_requests']} / {group['failed_requests']} | " + ' | '.join(values) + f" | {q['wer_percent']:.3f} | {q['sim_mean']:.3f} |")
    diagnostic_fps = ' and '.join(f"{g['performance']['throughput_frames_s']:.2f}" for g in data['diagnostic_groups'] if g['framework'] == 'vllm')
    readme_block = f'''<!-- latest-performance:start -->
Measured on **one B300**, using the same 1088 English requests and pre-encoded
references at each concurrency. Numeric rows are **medians of three complete runs**;
RTF is the median of run means. Two incomplete rows are marked below.
Versions pinned on 2026-10-16:
standalone {pin_links['standalone']}, vLLM-Omni {pin_links['vllm']}, SGLang-Omni {pin_links['sglang']}.

{table}

† vLLM C64 completed two runs ({diagnostic_fps} frames/s); its third run
stalled and reached the 300-second request timeout.

‡ SGLang C2 stopped after an HTTP 500 (no audio frames) in its second run;
that run had 1087 successes / 1 failure, and the third run was not attempted.
Neither incomplete row has a three-run estimate. C4–C64 used a new SGLang server
with the same configuration; see the session boundaries in the report.

`C` is the number of concurrent requests. Throughput counts returned audio frames
(2048 samples); TTFA measures time to first
PCM bytes. Complete groups include all 1088 requests and the final drain.
Standalone uses the [native B300 benchmark profile](configs/benchmark-native-b300.env).
Serving capacities are 64 / 128 / 64 and codec precision is BF16 / FP16 / FP16,
respectively.
[Configurations, quality scores, run ranges and raw evidence](docs/current-results.md).
<!-- latest-performance:end -->'''
    readme_path = args.repo / 'README.md'
    readme = readme_path.read_text()
    start = readme.index('<!-- latest-performance:start -->')
    end = readme.index('<!-- latest-performance:end -->') + len('<!-- latest-performance:end -->')
    readme = readme[:start] + readme_block + readme[end:]
    quality_table = ['| C | Standalone WER (%) / SIM | vLLM-Omni WER (%) / SIM | SGLang-Omni WER (%) / SIM |',
                     '|---:|---:|---:|---:|']
    range_table = ['| C | Standalone frames/s range | vLLM-Omni frames/s range | SGLang-Omni frames/s range |',
                   '|---:|---:|---:|---:|']
    for c in CONCURRENCY:
        quality = []; ranges = []
        for b in ORDER:
            if (b, c) not in lookup:
                label = 'Incomplete (2/3)' if b == 'vllm' else 'Incomplete (1 valid / 1 failed / 1 not run)'
                quality.append(label); ranges.append(label)
                continue
            m = lookup[(b, c)]['metrics']
            quality.append(f"{m['wer_percent']['median']:.3f} / {m['sim_mean']['median']:.3f}")
            ranges.append(f"{m['throughput_frames_s']['min']:.2f}–{m['throughput_frames_s']['max']:.2f}")
        quality_table.append(f'| {c} | ' + ' | '.join(quality) + ' |')
        range_table.append(f'| {c} | ' + ' | '.join(ranges) + ' |')
    versions = ['| Framework | Source | Base runtime | Torch | FlashInfer |', '|---|---|---|---|---|']
    for b in ORDER:
        deps = data['recorded_dependency_versions'][b]
        base = deps.get({'standalone': 'drama3-b300', 'vllm': 'vllm', 'sglang': 'sglang'}[b], 'dedicated Python engine')
        versions.append(f"| {NAMES[b]} | {pin_links[b]} | {base} | {deps.get('torch', 'recorded in preparation')} | {deps.get('flashinfer-python', 'recorded in preparation')} |")
    report = f'''# Matched three-framework B300 comparison

Sources were pinned on **2026-10-16**. This report replaces the README's earlier
mixture of old full-matrix measurements and newer standalone C64 observations.
All three frameworks use the same 1088 pre-encoded-reference requests per planned
run. The vLLM C64 third repetition stalled, and SGLang C2 stopped after an HTTP
500 in its second repetition. Both rows remain incomplete; neither is replaced
with an older version or an aggregate from fewer than three successful runs.
The SGLang adapter was corrected to pass its supported `references[].vq_codes`
field; this is a benchmark-client correction, not a change to the model.

Optimization remains paused. These are descriptive measurements of the existing
implementations, not a claim that the historical all-concurrency 1.5× target was
achieved. [Historical results](historical-results-20261016.md) retain their original
versions and measurement boundaries.

## Versions and controls

{chr(10).join(versions)}

The upstream source commits were the fetched main heads at preparation time;
the standalone runtime is frozen at the client-fix commit. Existing final-path
environments were reused without dependency upgrades. The installed vLLM-Omni
distribution metadata still carries an older editable-build version; CPU imports
and live launch records bind its actual modules to the new pinned worktree.
[Preparation and exact dependency versions]({EVIDENCE}/files/preparation.json).
SGLang's native `sm_103` dispatch uses PyTorch codebook sampling and disables
fused tower layers; its installed MoE kernels report generic configurations
because B300-specific tuning files are absent. These choices were recorded,
not overridden. The comparison does not establish each framework's performance
ceiling over all possible dependency versions and tuning configurations.

- Same physical B300: GPU6, UUID `{protocol['gpu_uuid']}`; OMP=8 and CPU affinity 0–239.
- Same Drama 3 checkpoint `{Path(protocol['checkpoint']).name}` and Fish Speech `{protocol['fish_source_sha']}`.
- Same ordered English Seed-TTS corpus, SHA-256 `{protocol['corpus_sha256']}`; all reference-code integers were checked across the three public adapters.
- Native FP8 experts and selected top-8 routes retained. Standalone uses [this profile](../configs/benchmark-native-b300.env), 64 slots, BF16 GDN/codec and asynchronous stateful DAC. vLLM uses its native two-stage `fish_omni.yaml`, private MPS, capacity 128 and FP16 stateful codec. SGLang uses capacity 64 and its native FP16 window codec. These are framework implementations, not identical operator configurations.
- Temperature 1, top-p 0.9, top-k 30, maximum 500 frames. Fresh TCP for every request, no retries. Before each concurrency: max(8,C) warmup requests, outside measured time. Each framework initially planned one server for 21 groups. vLLM stopped in its last group. SGLang stopped in C2; only its unattempted C4–C64 groups used a new server, with identical parameters and separate warmups. That restart resets serving history and can affect RNG state; this is not one continuous SGLang matrix session.
- A separate C64 feasibility run and full quality pass preceded the matrix. Feasibility values are excluded from the table. Preparation, process-to-readiness, warmup, measurement and scoring remain separate in the logs.

[Fixed protocol]({EVIDENCE}/files/protocol.json) · [Recovery matrix protocol]({EVIDENCE}/files/recovery-attempt-01/matrix-protocol.json) · [Post-vLLM continuation]({EVIDENCE}/files/interrupted-matrix-report-01/protocol.json) · [Remaining SGLang protocol]({EVIDENCE}/files/remaining-sglang-report-01/protocol.json) · [Machine-readable summary](current-results.json)

The original matrix stopped after seven complete groups when a whole-machine
GPU-monitor query exceeded its 10-second timeout. Those groups and the interrupted
output remain archived and are **excluded** from this table. The recovery attempt
restarted all 21 standalone groups and then started vLLM. SGLang followed under
the continuation described below. The recovery attempt's only controller change was UUID-scoped monitoring with a 30-second timeout; any timeout
still stopped the run. Model sources, dependencies and benchmark settings were
unchanged. [Stop audit]({EVIDENCE}/files/matrix-stop-audit.json).

## Interrupted vLLM C64 repetition

The final vLLM group, `c64-r2`, did not finish. Its last normal service log entries
were recorded at 15:28:38 UTC. At approximately 15:33:39, server cancellation
records appeared after the frozen
300-second client deadline, followed by stage-resource-release timeouts. The
controller was then interrupted and cleaned only its own processes; GPU6 was
empty afterwards. The root cause has not been established, and this group was
not retried. [Failure audit]({EVIDENCE}/files/recovery-attempt-01/vllm-stop-audit.json).

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
[Failure audit]({EVIDENCE}/files/interrupted-matrix-report-01/sglang-stop-audit.json).

All saved audio was scored, including the 1087 outputs from the invalid group.
The failed request retains its error and is not treated as a scored output. Its partial
scores are diagnostics, not a quality score for all 1088 requested samples.
Neither the failed request nor any completed C1/C2 group was retried.
A separately frozen continuation measured only the previously unattempted
C4, C8, C16, C32 and C64 groups, three times each, then scored all their outputs.
Sources, inputs, capacities, sampling and warmup policy remained unchanged.
The new SGLang process is an explicit session boundary; its serving/RNG history
differs from the interrupted process, and each row records its source session.

The following individual observations are **diagnostic only**, not formal
three-run estimates. For the failed SGLang group, throughput uses returned audio
and the whole group makespan; TTFA/RTF and quality cover its 1087 successful
outputs only. They are not substituted for the missing C2 row:

{chr(10).join(diagnostic_table)}

## Performance

{table}

`—†` and `—‡` denote the incomplete vLLM C64 and SGLang C2 rows described above. Each numeric row is the independent median of three complete 1088-request groups.
Throughput = total returned PCM samples / 2048 / full group makespan.
TTFA p50/p95 = per-run percentiles of dispatch-to-first-PCM time, then medians
across runs. RTF = each request's dispatch-to-last-PCM time / audio duration;
the table reports the median of the three run means. Audio is mono PCM16 at
44.1 kHz. Initial fill, final drain and frame-capped outputs are included.
Raw PCM does not expose EOS reasons. Generated lengths and random-number
schedules differ; no equal-output or bitwise-equivalence claim is made.

Observed throughput ranges across the three repetitions (not confidence intervals):

{chr(10).join(range_table)}

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

{chr(10).join(quality_table)}

{matrix_quality_note}
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
in [the evidence archive]({EVIDENCE}/README.md). WAV audio and model weights stay
outside Git; each raw request record contains the returned PCM SHA-256.

The former unmatched SGLang 192-request measurements and older standalone
versions remain in [the historical archive](historical-results-20261016.md).
'''
    output = dict(data, as_of='2026-10-16', optimization_status='PAUSED_AT_USER_REQUEST',
                  document_generated_utc=datetime.now(timezone.utc).isoformat(), source_shas=pins,
                  corpus_sha256=protocol['corpus_sha256'],
                  comparison_scope=protocol['comparison_scope'],
                  standalone_profile='configs/benchmark-native-b300.env')
    if args.write:
        readme_path.write_text(readme)
        (args.repo / 'docs/current-results.md').write_text(report)
        (args.repo / 'docs/current-results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'ready': True, 'written': args.write, 'numeric_rows': len(rows), 'incomplete_rows': 2,
                      'standalone_c1_to_c32_within_quality_reporting_bounds': bounds,
                      'readme_lines': len(readme.splitlines())}))


if __name__ == '__main__':
    main()
