"""Summarize declared measurements while retaining both incomplete concurrency rows."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = OUT.parent
REPORT_ID = 'remaining-sglang-report-01'
RECOVERY = 'recovery-attempt-01'
INTERRUPTED = 'interrupted-matrix-report-01'
BACKENDS = ('standalone', 'vllm', 'sglang')
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
PERF = ('throughput_frames_s', 'ttfa_p50_ms', 'ttfa_p95_ms', 'rtf_mean')
STATUS = 'COMPLETE_DESCRIPTIVE_COMPARISON_WITH_INCOMPLETE_ROWS'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), 'invalid three-run values')
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def aggregate(report_dir):
    require(Path(report_dir).resolve() == OUT and OUT.name == REPORT_ID, 'explicit report directory differs')
    recovery, interrupted = ART / RECOVERY, ART / INTERRUPTED
    protocol = read(ART / 'protocol.json')
    require(protocol['concurrency'] == list(CONCURRENCY) and protocol['repeats'] == 3
            and protocol['requests'] == 1088 and set(protocol['frameworks']) == set(BACKENDS), 'base contract differs')
    bindings = [('protocol_sha256', ART / 'protocol.json'), ('manifest_sha256', ART / 'frozen-inputs.json'),
                ('matrix_protocol_sha256', recovery / 'matrix-protocol.json'), ('matrix_manifest_sha256', recovery / 'matrix-frozen-inputs.json'),
                ('continuation_protocol_sha256', interrupted / 'protocol.json'), ('continuation_manifest_sha256', interrupted / 'frozen-inputs.json'),
                ('remaining_protocol_sha256', OUT / 'protocol.json'), ('remaining_manifest_sha256', OUT / 'frozen-inputs.json')]
    hashes = {key: digest(path) for key, path in bindings}
    for key, path in bindings:
        if 'manifest' not in key:
            continue
        manifest = read(path)
        entries = manifest.get('files', manifest)
        require(entries, 'empty frozen manifest')
        for name, record in entries.items():
            require(Path(name).stat().st_size == record['bytes'] and digest(name) == record['sha256'], 'frozen input changed: ' + name)
    all_labels = [f'c{c}-r{r}' for c in CONCURRENCY for r in range(3)]
    segments = [
        ('standalone', 'standalone', recovery / 'matrix/standalone', recovery / 'matrix/standalone', all_labels, 4),
        ('vllm', 'vllm', recovery / 'matrix/vllm', interrupted / 'vllm-quality', all_labels[:-1], 6),
        ('sglang_saved', 'sglang', interrupted / 'sglang', OUT / 'existing-quality', all_labels[:5], 8),
        ('sglang_remaining', 'sglang', OUT / 'remaining', OUT / 'remaining', all_labels[6:], 8),
    ]
    client_hash = digest(ART / 'source-standalone/src/drama3/benchmark.py')
    performance, quality, owners, provenance, failed = {}, {}, {}, {}, []
    for segment, backend, service_root, result_root, labels, binding_count in segments:
        result = read(result_root / 'result.json')
        expected = {'vllm': 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY',
                    'sglang_saved': 'COMPLETE_VALID_SAVED_AUDIO_WITH_PRESERVED_FAILURE'}.get(segment, 'COMPLETE_VALID_SERVICE_AND_QUALITY')
        require(result['status'] == expected, segment + ' incomplete')
        expected_attempt = RECOVERY if segment == 'standalone' else INTERRUPTED if segment == 'vllm' else REPORT_ID
        require(result['attempt_id'] == expected_attempt, segment + ' attempt differs')
        for key, _ in bindings[:binding_count]:
            require(result[key] == hashes[key], segment + ' protocol identity differs: ' + key)
        launch = read(service_root / 'service/launch.json')
        require(launch['source_sha'] == protocol['source_shas'][backend] and not launch['source_status']
                and launch['benchmark_client_sha256'] == client_hash, segment + ' source/client differs')
        pa, qa = result['service'], result['quality']
        for kind, audit in [('performance', pa), ('quality', qa)]:
            audit_status = ('VALIDATED_WITH_KNOWN_PARTIAL_FAILURE' if segment == 'sglang_saved'
                            else 'PASS_COMPLETED_GROUPS_ONLY' if segment == 'vllm' and kind == 'performance' else 'PASS')
            require(audit['status'] == audit_status and audit['groups_checked'] == len(labels)
                    and audit['requests_checked'] == 1088 * len(labels), segment + ' incomplete ' + kind)
            require(len(audit['groups']) == len(labels) and {g['label'] for g in audit['groups']} == set(labels), segment + ' group coverage differs')
        if segment == 'vllm':
            require(pa['planned_groups'] == 21 and pa['interrupted_group'] == 'c64-r2'
                    and pa['full_matrix_valid'] is False, 'vLLM failure scope missing')
        if segment == 'sglang_saved':
            for audit in (pa, qa):
                require(audit['full_matrix_valid'] is False and audit['complete_groups'] == 4
                        and audit['partial_groups'] == 1 and audit['failed_records'] == 1,
                        'SGLang partial scope missing')
            require(pa['successful_audio_checked'] == qa['scored_requests'] == 5439
                    and pa['all_successful_wav_pcm_hashes_verified']
                    and qa['all_successful_sim_records_present'] and qa['known_failure_sim'] is None,
                    'SGLang successful-audio coverage or null failure score missing')
        else:
            require(pa['all_wav_pcm_hashes_verified'] and qa['all_sim_records_present'], segment + ' audio/quality audit missing')
        require(qa['unfiltered_wer'], segment + ' unfiltered quality audit missing')
        require(pa['corpus_sha256'] == protocol['corpus_sha256']
                and pa['summary_sha256'] == digest(service_root / 'service/benchmark/summary.json')
                and qa['quality_sha256'] == digest(result_root / 'quality.json'), segment + ' raw audit drift')
        pg, qg = {g['label']: g for g in pa['groups']}, {g['label']: g for g in qa['groups']}
        quality_records = read(result_root / 'quality.json')['groups']
        for label in labels:
            raw_path = service_root / 'service/benchmark' / (label + '.jsonl')
            require(pg[label]['jsonl_sha256'] == digest(raw_path), segment + ' raw records changed')
            require(pg[label]['requests'] == qg[label]['requests'] == 1088, segment + ' short source group')
            require(all(math.isfinite(pg[label][k]) for k in PERF)
                    and all(math.isfinite(qg[label][k]) for k in ('wer_percent', 'sim_mean')), segment + ' nonfinite metric')
            records = quality_records[label]['records']
            require(len(records) == len({r['request_id'] for r in records}) == 1088, segment + ' quality records missing')
            partial = segment == 'sglang_saved' and label == 'c2-r1'
            scored = [r for r in records if r['is_success']]
            require(len(scored) == (1087 if partial else 1088)
                    and all(r['speaker_similarity'] is not None and math.isfinite(r['speaker_similarity']) for r in scored), segment + ' score coverage differs')
            if partial:
                failures = [r for r in records if not r['is_success']]
                require(len(failures) == 1 and failures[0]['speaker_similarity'] is None
                        and quality_records[label]['complete'] is False, 'failed output must remain unscored')
            require((backend, label) not in performance, 'duplicate group across server sessions')
            performance[(backend, label)], quality[(backend, label)], owners[(backend, label)] = pg[label], qg[label], segment
        service_runner = read(service_root / 'service-runner/result.json')
        quality_runner = read(result_root / 'quality-runner/result.json')
        was_stopped = segment in ('vllm', 'sglang_saved')
        require(service_runner['status'] == ('FAIL' if was_stopped else 'PASS') and service_runner['post_gpu'] == [], segment + ' service cleanup incomplete')
        require(quality_runner['status'] == 'PASS' and quality_runner['post_gpu'] == [], segment + ' scoring cleanup incomplete')
        if was_stopped:
            original = read(service_root / 'result.json')
            require(original['status'] == 'STOPPED_FAILURE' and service_runner.get('error'), segment + ' failure not preserved')
            require(result['failed_service_result_sha256'] == digest(service_root / 'result.json')
                    and result['failed_runner_result_sha256'] == digest(service_root / 'service-runner/result.json'), segment + ' failure identity differs')
            failed.append({'framework': backend, 'label': 'c64-r2' if backend == 'vllm' else 'c2-r1',
                           'source_result': str((service_root / 'result.json').relative_to(ART)),
                           'source_result_sha256': digest(service_root / 'result.json'),
                           'runner_result_sha256': digest(service_root / 'service-runner/result.json'),
                           'request_failures': None if backend == 'vllm' else 1})
        provenance[segment] = {'framework': backend, 'source_sha': launch['source_sha'], 'launch': launch,
                               'service_directory': str(service_root.relative_to(ART)), 'result_directory': str(result_root.relative_to(ART)),
                               'result_sha256': digest(result_root / 'result.json'), 'performance_audit': pa, 'quality_audit': qa}
    rows, diagnostics = [], []
    for backend in BACKENDS:
        for c in CONCURRENCY:
            labels = [f'c{c}-r{r}' for r in range(3)]
            incomplete = (backend == 'vllm' and c == 64) or (backend == 'sglang' and c == 2)
            if incomplete:
                for label in labels[:2]:
                    partial = backend == 'sglang' and label == 'c2-r1'
                    diagnostics.append({'framework': backend, 'concurrency': c, 'label': label, 'formal_three_run_row': False,
                                        'valid_complete_group': not partial, 'attempted_requests': 1088,
                                        'successful_requests': 1087 if partial else 1088, 'failed_requests': 1 if partial else 0,
                                        'scored_requests': 1087 if partial else 1088,
                                        'source_segment': owners[(backend, label)],
                                        'performance': performance[(backend, label)], 'quality': quality[(backend, label)]})
                continue
            segment = owners[(backend, labels[0])]
            require(all(owners[(backend, label)] == segment for label in labels), 'formal row crosses server sessions')
            metrics = {k: stats([performance[(backend, label)][k] for label in labels]) for k in PERF}
            metrics.update({k: stats([quality[(backend, label)][k] for label in labels]) for k in ('wer_percent', 'sim_mean')})
            rows.append({'framework': backend, 'concurrency': c, 'repeats': 3, 'requests_per_repeat': 1088, 'metrics': metrics,
                         'source_segment': segment, 'server_session': provenance[segment]['service_directory'],
                         'source_result_directory': provenance[segment]['result_directory']})
    for row in rows:
        backend, c = row['framework'], row['concurrency']
        if c == 64:
            row['quality_vs_vllm'] = {'available': False, 'reason': 'vLLM C64 lacks its third completed run'}
            continue
        base = next(x for x in rows if x['framework'] == 'vllm' and x['concurrency'] == c)
        dw = row['metrics']['wer_percent']['median'] - base['metrics']['wer_percent']['median']
        ds = row['metrics']['sim_mean']['median'] - base['metrics']['sim_mean']['median']
        paired = [{'repeat': r, 'wer_delta_pp': quality[(backend, f'c{c}-r{r}')]['wer_percent'] - quality[('vllm', f'c{c}-r{r}')]['wer_percent'],
                   'sim_delta': quality[(backend, f'c{c}-r{r}')]['sim_mean'] - quality[('vllm', f'c{c}-r{r}')]['sim_mean']} for r in range(3)]
        row['quality_vs_vllm'] = {'available': True, 'wer_median_delta_pp': dw, 'sim_median_delta': ds,
                                  'median_within_reporting_bounds': dw <= 1 and ds >= -2, 'paired_repeat_differences': paired,
                                  'all_repeats_within_reporting_bounds': all(x['wer_delta_pp'] <= 1 and x['sim_delta'] >= -2 for x in paired)}
    require(len(rows) == 19 and len(diagnostics) == 4 and len(performance) == 61, 'wrong final coverage')
    preparation = read(ART / 'preparation.json')
    return {'status': STATUS, 'optimization_acceptance_evaluated': False, 'complete_matrix': False,
            'report_id': REPORT_ID, 'aggregation': 'Three complete repetitions per formal row only; ranges are observed min/max. No retry, invalid-group filtering into valid metrics, or matching RNG claim.',
            'performance_groups': 60, 'quality_groups': 60, 'complete_group_requests': 65280,
            'partial_performance_groups': 1, 'partial_quality_groups': 1, 'partial_scored_requests': 1087,
            'total_scored_requests': 66367, 'expected_groups': 63, 'formal_rows': 19, 'rows': rows,
            'incomplete_rows': [{'framework': 'vllm', 'concurrency': 64, 'expected_repeats': 3, 'completed_repeats': 2, 'failed_label': 'c64-r2', 'metrics': None},
                                {'framework': 'sglang', 'concurrency': 2, 'expected_repeats': 3, 'completed_repeats': 1, 'failed_label': 'c2-r1', 'not_run_label': 'c2-r2', 'metrics': None}],
            'diagnostic_groups': diagnostics, 'failed_groups': failed,
            'not_run_groups': [{'framework': 'sglang', 'label': 'c2-r2', 'status': 'NOT_RUN', 'reason': 'No retry or completion of failed C2 row authorized'}],
            'excluded_from_formal_metrics': ['original seven-group monitor-failed attempt', 'all feasibility groups', 'vLLM C64 and SGLang C2 incomplete rows'],
            'sglang_server_lifecycles': {'count': 2, 'old': provenance['sglang_saved']['service_directory'], 'remaining': provenance['sglang_remaining']['service_directory'],
                                        'scope': 'Fresh server and capture/RNG/warmup boundary before C4..C64; no shared 21-group server claim'},
            'quality_reporting_bounds': {'wer_increase_pp': 1, 'sim_decrease': 2, 'filters_results': False, 'standalone_available_concurrency': [1, 2, 4, 8, 16, 32]},
            **hashes, 'client_sha256': client_hash, 'generator_sha256': digest(__file__),
            'recorded_dependency_versions': {b: preparation['imports'][b]['packages'] for b in BACKENDS},
            'preparation_sha256': digest(ART / 'preparation.json'), 'provenance': provenance}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    data = aggregate(args.report_dir)
    with (OUT / 'matrix-summary.json').open('x') as output:
        output.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
