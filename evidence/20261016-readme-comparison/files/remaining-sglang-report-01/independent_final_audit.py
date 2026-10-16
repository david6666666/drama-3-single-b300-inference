"""Independent CPU audit of the segmented campaign; no frozen helper imports."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = OUT.parent
CS = (1, 2, 4, 8, 16, 32, 64)
PERF = ('throughput_frames_s', 'ttfa_p50_ms', 'ttfa_p95_ms', 'rtf_mean')
LABELS = [f'c{c}-r{r}' for c in CS for r in range(3)]
CHECKS = []


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, name):
    if not ok:
        raise AssertionError(name)


def checked(ok, name):
    require(ok, name)
    CHECKS.append({'name': name, 'passed': True})


def close(a, b, name):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), name)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def word_distance(ref, hyp):
    a, b = ref.split(), hyp.split()
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def performance(bench, saved):
    label = f"c{saved['concurrency']}-r{saved['repeat']}"
    rows = [json.loads(line) for line in (bench / (label + '.jsonl')).read_text().splitlines() if line.strip()]
    require(len(rows) == len({r['request_id'] for r in rows}) == 1088, label + ': record count')
    require({r['request_id'] for r in rows} == {f'{label}-{i:05d}' for i in range(1088)}, label + ': exact request IDs')
    successes = [r for r in rows if r['success']]
    require(saved['completed'] == len(successes) and saved['failed'] == 1088 - len(successes)
            and saved['valid'] is (len(successes) == 1088), label + ': completeness')
    observed = max(r['dispatch_s'] + r['completion_s'] for r in rows) - min(r['dispatch_s'] for r in rows)
    require(saved['makespan_s'] + 1e-7 >= observed, label + ': full completion tail')
    rtfs = [r['last_audio_s'] / (r['audio_samples'] / 44100) for r in successes]
    for row, rtf in zip(successes, rtfs, strict=True):
        close(row['rtf'], rtf, row['request_id'] + ': independent RTF')
    metrics = {'throughput_frames_s': sum(r['audio_samples'] for r in successes) / 2048 / saved['makespan_s'],
               'ttfa_p50_ms': percentile([r['ttfa_s'] for r in successes], .5) * 1000,
               'ttfa_p95_ms': percentile([r['ttfa_s'] for r in successes], .95) * 1000,
               'rtf_mean': statistics.mean(rtfs)}
    for key in PERF:
        close(metrics[key], saved[key], label + ': performance ' + key)
    return label, rows, metrics


def verify_pcm(row, bench):
    rid = row['request_id']
    with wave.open(str(bench / (rid + '.wav')), 'rb') as wav:
        require((wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) == (1, 44100, 2, 'NONE'), rid + ': WAV format')
        require(wav.getnframes() == row['audio_samples'], rid + ': PCM frames')
        pcm = wav.readframes(wav.getnframes())
    require(hashlib.sha256(pcm).hexdigest() == row['audio_sha256'], rid + ': PCM hash')
    require(sum(row['chunk_bytes']) == len(pcm) == row['audio_samples'] * 2, rid + ': PCM transport')
    arrivals = row['chunk_arrivals_s']
    require(arrivals == sorted(arrivals) and len(arrivals) == len(row['chunk_bytes']) == row['transport_chunks'] > 0, rid + ': chunk sequence')
    require(0 <= row['ttfa_s'] <= row['last_audio_s'] <= row['completion_s'], rid + ': timing order')
    close(row['ttfa_s'], arrivals[0], rid + ': first PCM')
    close(row['last_audio_s'], arrivals[-1], rid + ': last PCM')
    close(row['rtf'], row['last_audio_s'] / (row['audio_samples'] / 44100), rid + ': RTF')
    close(row['acoustic_frames'], row['audio_samples'] / 2048, rid + ': acoustic frames')
    return len(pcm)


def verify_quality(group, rows, samples, *, bench=None):
    records = group['records']
    require(len(records) == 1088, 'quality count')
    scored, pcm_bytes, distances = [], 0, 0
    for row, record in zip(rows, records, strict=True):
        rid = row['request_id']
        sample = samples[int(rid.rsplit('-', 1)[1])]
        require(record['request_id'] == rid and record['sample_id'] == row['sample_id'] == sample['id']
                and record['target_text'] == sample['text'], rid + ': corpus identity/order')
        if not row['success']:
            require(rid == 'c2-r1-00351' and row['http_status'] == 500
                    and 'Drama 3 generated no audio frames' in row['error'], rid + ': original failure')
            require(record['is_success'] is False and record['error'] == row['error'] and record['speaker_similarity'] is None, rid + ': failure score')
            require(all(record[k] == 0 for k in ('wer', 'substitutions', 'deletions', 'insertions', 'hits'))
                    and all(record[k] == '' for k in ('whisper_text', 'ref_norm', 'hyp_norm')), rid + ': failure placeholders')
            continue
        require(record['is_success'] is True and not record['error'] and math.isfinite(record['speaker_similarity']), rid + ': score')
        if bench is not None:
            pcm_bytes += verify_pcm(row, bench)
            require(word_distance(record['ref_norm'], record['hyp_norm']) == record['substitutions'] + record['deletions'] + record['insertions'], rid + ': independent edit distance')
            distances += 1
        words = len(record['ref_norm'].split())
        require(words == record['hits'] + record['substitutions'] + record['deletions'], rid + ': reference words')
        errors = record['substitutions'] + record['deletions'] + record['insertions']
        close(record['wer'], errors / words if words else float(record['insertions']), rid + ': WER')
        scored.append(record)
    skipped = 1088 - len(scored)
    require(group['complete'] is (skipped == 0) and group['wer']['evaluated'] == group['similarity']['evaluated'] == len(scored)
            and group['wer']['skipped'] == group['similarity']['skipped'] == skipped, 'quality coverage')
    errors = sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in scored)
    words = sum(len(r['ref_norm'].split()) for r in scored)
    values = [r['wer'] for r in scored]
    metrics = {'wer_corpus': errors / words, 'wer_per_sample_mean': statistics.mean(values),
               'wer_per_sample_median': statistics.median(values), 'wer_per_sample_std': statistics.pstdev(values),
               'wer_per_sample_p95': percentile(values, .95), 'wer_per_sample_max': max(values)}
    for key, value in metrics.items():
        close(value, group['wer'][key], 'unfiltered quality ' + key)
    sim = statistics.mean(r['speaker_similarity'] for r in scored)
    close(sim, group['similarity']['mean'], 'unfiltered SIM')
    return {'wer_percent': errors / words * 100, 'sim_mean': sim}, pcm_bytes, distances, len(scored)


def complete_audit():
    summary = read(OUT / 'matrix-summary.json')
    protocol = read(ART / 'protocol.json')
    binding_paths = [('protocol_sha256', ART / 'protocol.json'), ('manifest_sha256', ART / 'frozen-inputs.json'),
                     ('matrix_protocol_sha256', ART / 'recovery-attempt-01/matrix-protocol.json'), ('matrix_manifest_sha256', ART / 'recovery-attempt-01/matrix-frozen-inputs.json'),
                     ('continuation_protocol_sha256', ART / 'interrupted-matrix-report-01/protocol.json'), ('continuation_manifest_sha256', ART / 'interrupted-matrix-report-01/frozen-inputs.json'),
                     ('remaining_protocol_sha256', OUT / 'protocol.json'), ('remaining_manifest_sha256', OUT / 'frozen-inputs.json')]
    bindings = {k: sha(p) for k, p in binding_paths}
    checked(all(summary[k] == v for k, v in bindings.items()), 'summary_eight_bindings')
    counts = {}
    for key, path in binding_paths:
        if 'manifest' not in key:
            continue
        manifest = read(path)
        manifest = manifest.get('files', manifest)
        checked(all(Path(p).stat().st_size == v['bytes'] and sha(Path(p)) == v['sha256'] for p, v in manifest.items()), 'unchanged_' + key)
        counts[key] = len(manifest)
    checked(list(counts.values()) == [7313, 16, 51, 33], 'all_four_manifest_counts')
    samples = [json.loads(line) for line in Path(protocol['corpus']).read_text().splitlines() if line.strip()]
    checked(len(samples) == 1088 and sha(Path(protocol['corpus'])) == protocol['corpus_sha256'], 'corpus_identity')
    for backend in ('standalone', 'vllm', 'sglang'):
        source = ART / ('source-' + backend)
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        checked(revision == protocol['source_shas'][backend] and not dirty, backend + '_source_revision_clean')
    segments = [('standalone', 'standalone', 'recovery-attempt-01/matrix/standalone', 'recovery-attempt-01/matrix/standalone', LABELS, 4, 'COMPLETE_VALID_SERVICE_AND_QUALITY'),
                ('vllm', 'vllm', 'recovery-attempt-01/matrix/vllm', 'interrupted-matrix-report-01/vllm-quality', LABELS[:-1], 6, 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY'),
                ('sglang_saved', 'sglang', 'interrupted-matrix-report-01/sglang', 'remaining-sglang-report-01/existing-quality', LABELS[:5], 8, 'COMPLETE_VALID_SAVED_AUDIO_WITH_PRESERVED_FAILURE'),
                ('sglang_remaining', 'sglang', 'remaining-sglang-report-01/remaining', 'remaining-sglang-report-01/remaining', LABELS[6:], 8, 'COMPLETE_VALID_SERVICE_AND_QUALITY')]
    measured, owners, evidence = {}, {}, {}
    pcm_bytes = distances = audio_scores = raw_comparisons = 0
    for segment, backend, service_rel, result_rel, labels, binding_count, status in segments:
        service, result_dir = ART / service_rel, ART / result_rel
        result = read(result_dir / 'result.json')
        checked(result['status'] == status and all(result[k] == bindings[k] for k, _ in binding_paths[:binding_count]), segment + '_completion_bindings')
        launch = read(service / 'service/launch.json')
        checked(launch['source_sha'] == protocol['source_shas'][backend] and launch['source_status'] == ''
                and launch['benchmark_client_sha256'] == summary['client_sha256'], segment + '_launch_identity')
        checked(read(result_dir / 'quality-runner/result.json')['status'] == 'PASS'
                and read(result_dir / 'quality-runner/result.json')['post_gpu'] == [], segment + '_quality_cleanup')
        stopped = segment in ('vllm', 'sglang_saved')
        sr = read(service / 'service-runner/result.json')
        checked(sr['status'] == ('FAIL' if stopped else 'PASS') and sr['post_gpu'] == [], segment + '_service_cleanup')
        if stopped:
            checked(read(service / 'result.json')['status'] == 'STOPPED_FAILURE'
                    and result['failed_service_result_sha256'] == sha(service / 'result.json')
                    and result['failed_runner_result_sha256'] == sha(service / 'service-runner/result.json'), segment + '_original_failure_retained')
        bench = service / 'service/benchmark'
        raw_summary, scored = read(bench / 'summary.json'), read(result_dir / 'quality.json')
        checked([f"c{s['concurrency']}-r{s['repeat']}" for s in raw_summary] == labels and list(scored['groups']) == labels, segment + '_exact_group_coverage')
        checked(summary['provenance'][segment]['service_directory'] == service_rel
                and summary['provenance'][segment]['result_directory'] == result_rel
                and summary['provenance'][segment]['source_sha'] == launch['source_sha']
                and summary['provenance'][segment]['launch'] == launch, segment + '_summary_session_provenance')
        for saved in raw_summary:
            label, rows, metrics = performance(bench, saved)
            partial = backend == 'sglang' and label == 'c2-r1'
            require(sum(r['success'] for r in rows) == 1088 - int(partial), segment + label + ': unexpected failure')
            qm, size, nd, count = verify_quality(scored['groups'][label], rows, samples, bench=bench if segment == 'sglang_remaining' else None)
            pcm_bytes += size
            distances += nd
            audio_scores += count
            raw_comparisons += len(PERF)
            require((backend, label) not in measured, 'duplicate group across server sessions')
            measured[backend, label] = dict(metrics, **qm)
            owners[backend, label] = (segment, service_rel, result_rel)
        for path in (result_dir / 'result.json', result_dir / 'quality.json', bench / 'summary.json', service / 'service/launch.json'):
            evidence[str(path.relative_to(ART))] = sha(path)
    checked(distances == 16320, 'all16320_new_PCM_and_normalized_edit_distances')
    checked(audio_scores == 66367 and len(measured) == 61, '60complete_groups_plus1087partial_scores')
    checked(raw_comparisons == 244, 'all61_saved_groups_independent_four_metrics')
    expected_rows = {(b, c) for b in ('standalone', 'vllm', 'sglang') for c in CS} - {('vllm', 64), ('sglang', 2)}
    checked(len(summary['rows']) == 19 and {(r['framework'], r['concurrency']) for r in summary['rows']} == expected_rows, '19_formal_three_run_rows')
    comparisons = []
    for row in summary['rows']:
        backend, c = row['framework'], row['concurrency']
        triple = [measured[backend, f'c{c}-r{r}'] for r in range(3)]
        owner = owners[backend, f'c{c}-r0']
        require(all(owners[backend, f'c{c}-r{r}'] == owner for r in range(3)), 'row crosses server lifecycles')
        require((row['source_segment'], row['server_session'], row['source_result_directory']) == owner, 'formal row provenance')
        for key in PERF + ('wer_percent', 'sim_mean'):
            values = [x[key] for x in triple]
            for operation, value in [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]:
                close(row['metrics'][key][operation], value, backend + str(c) + key + operation)
        q = row['quality_vs_vllm']
        if c == 64:
            require(q['available'] is False, 'no vLLM64 complete quality comparison')
            continue
        base = [measured['vllm', f'c{c}-r{r}'] for r in range(3)]
        dw = statistics.median(x['wer_percent'] for x in triple) - statistics.median(x['wer_percent'] for x in base)
        ds = statistics.median(x['sim_mean'] for x in triple) - statistics.median(x['sim_mean'] for x in base)
        close(q['wer_median_delta_pp'], dw, 'quality median delta WER')
        close(q['sim_median_delta'], ds, 'quality median delta SIM')
        repeat_bounds = []
        for r in range(3):
            wr, sr = triple[r]['wer_percent'] - base[r]['wer_percent'], triple[r]['sim_mean'] - base[r]['sim_mean']
            close(q['paired_repeat_differences'][r]['wer_delta_pp'], wr, 'repeat WER delta')
            close(q['paired_repeat_differences'][r]['sim_delta'], sr, 'repeat SIM delta')
            repeat_bounds.append(wr <= 1 and sr >= -2)
        require(q['available'] and q['median_within_reporting_bounds'] == (dw <= 1 and ds >= -2)
                and q['all_repeats_within_reporting_bounds'] == all(repeat_bounds), 'fixed quality bounds')
        if backend == 'standalone':
            comparisons.append({'concurrency': c, 'wer_median_delta_pp': dw, 'sim_median_delta': ds, 'all_repeats_within_reporting_bounds': all(repeat_bounds)})
    checked(True, 'all19_row_medians_ranges_sessions_and_available_quality_bounds')
    require(len(summary['incomplete_rows']) == 2 and {(r['framework'], r['concurrency']) for r in summary['incomplete_rows']} == {('vllm', 64), ('sglang', 2)}
            and all(r['metrics'] is None for r in summary['incomplete_rows']), 'two incomplete rows')
    expected_diagnostics = {('vllm', 'c64-r0'), ('vllm', 'c64-r1'), ('sglang', 'c2-r0'), ('sglang', 'c2-r1')}
    require(len(summary['diagnostic_groups']) == 4 and {(g['framework'], g['label']) for g in summary['diagnostic_groups']} == expected_diagnostics, 'four diagnostics')
    for group in summary['diagnostic_groups']:
        key = group['framework'], group['label']
        partial = key == ('sglang', 'c2-r1')
        require(group['source_segment'] == owners[key][0], 'diagnostic source session')
        require(group['attempted_requests'] == 1088 and group['successful_requests'] == group['scored_requests'] == 1088 - int(partial)
                and group['failed_requests'] == int(partial) and group['valid_complete_group'] is (not partial)
                and group['formal_three_run_row'] is False, 'diagnostic denominators')
        for metric in PERF:
            close(group['performance'][metric], measured[key][metric], 'diagnostic performance')
        for metric in ('wer_percent', 'sim_mean'):
            close(group['quality'][metric], measured[key][metric], 'diagnostic quality')
    checked(True, 'two_incomplete_rows_four_diagnostics_no_partial_imputation')
    checked(summary['status'] == 'COMPLETE_DESCRIPTIVE_COMPARISON_WITH_INCOMPLETE_ROWS' and summary['complete_matrix'] is False
            and summary['optimization_acceptance_evaluated'] is False and summary['performance_groups'] == summary['quality_groups'] == 60
            and summary['complete_group_requests'] == 65280 and summary['partial_scored_requests'] == 1087
            and summary['total_scored_requests'] == 66367, 'descriptive_not_full_matrix_status_and_denominators')
    checked(summary['sglang_server_lifecycles']['count'] == 2
            and summary['sglang_server_lifecycles']['old'] == 'interrupted-matrix-report-01/sglang'
            and summary['sglang_server_lifecycles']['remaining'] == 'remaining-sglang-report-01/remaining'
            and summary['not_run_groups'][0]['label'] == 'c2-r2'
            and summary['not_run_groups'][0]['status'] == 'NOT_RUN', 'two_SG_lifecycles_and_C2r2_NOT_RUN')
    expected_failures = {('vllm', 'c64-r2'), ('sglang', 'c2-r1')}
    checked({(g['framework'], g['label']) for g in summary['failed_groups']} == expected_failures, 'both_upstream_failures_retained')
    report = {'status': 'INDEPENDENT_FINAL_COMPARISON_REVIEW_PASS', 'complete_matrix': False,
              'optimization_acceptance_evaluated': False, 'checks': CHECKS, 'frozen_manifest_counts': counts,
              'new_SG_groups': 15, 'new_PCM_files_verified': distances, 'new_PCM_bytes_verified': pcm_bytes,
              'new_normalized_word_edit_distances_recomputed': distances, 'all_saved_groups_metric_comparisons': raw_comparisons,
              'all_successful_quality_records_aggregated': audio_scores, 'formal_rows': 19, 'incomplete_rows': 2, 'diagnostic_groups': 4,
              'standalone_C1_C32_quality_vs_vllm': comparisons, 'bindings': bindings,
              'summary_sha256': sha(OUT / 'matrix-summary.json'), 'evidence_sha256': evidence, 'script_sha256': sha(Path(__file__)),
              'limits': ['CPU/read-only audit; no inference, text normalization or GPU operations.',
                         '16320 new PCM files and normalized word edit distances independently checked; prior PCM/edit-distance audits remain separate hash-bound evidence.',
                         'C2 partial metrics/quality describe1087 successful outputs from1088 attempts, never a complete workload.',
                         'No complete63-group matrix, C64 vLLM quality verdict, independent matching RNG, or optimization acceptance claim.',
                         'Archive and rendered repository documents are not inspected by this script.']}
    with (OUT / 'independent-final-review.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(CHECKS), 'new_PCM_bytes': pcm_bytes,
                      'standalone_quality_bounds': comparisons, 'review_sha256': sha(OUT / 'independent-final-review.json')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    if args.complete:
        complete_audit()
        return
    bench = OUT / 'remaining/service/benchmark'
    saved = read(bench / 'summary.json')
    require(all(x['concurrency'] in (4, 8, 16, 32, 64) and x['valid'] for x in saved), 'unexpected incomplete saved group')
    metrics = []
    for group in saved:
        label, rows, values = performance(bench, group)
        metrics.append({'label': label, 'requests': len(rows), **values})
    print(json.dumps({'status': 'READ_ONLY_COMPLETED_PERFORMANCE_INSPECTION', 'groups': metrics, 'files_written': False}, indent=2))


if __name__ == '__main__':
    main()
