"""CPU-only descriptive report with one explicit incomplete vLLM C64 row."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
ART = OUT.parent
REPORT_ID = 'interrupted-matrix-report-01'
RECOVERY = 'recovery-attempt-01'
BACKENDS = ('standalone', 'vllm', 'sglang')
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
PERF = ('throughput_frames_s', 'ttfa_p50_ms', 'ttfa_p95_ms', 'rtf_mean')
STATUS = 'COMPLETE_DESCRIPTIVE_COMPARISON_WITH_INCOMPLETE_ROW'


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
    recovery = ART / RECOVERY
    protocol = read(ART / 'protocol.json')
    require(protocol['concurrency'] == list(CONCURRENCY) and protocol['repeats'] == 3
            and protocol['requests'] == 1088 and set(protocol['frameworks']) == set(BACKENDS), 'base contract differs')
    hashes = {'protocol_sha256': digest(ART / 'protocol.json'),
              'manifest_sha256': digest(ART / 'frozen-inputs.json'),
              'matrix_protocol_sha256': digest(recovery / 'matrix-protocol.json'),
              'matrix_manifest_sha256': digest(recovery / 'matrix-frozen-inputs.json'),
              'continuation_protocol_sha256': digest(OUT / 'protocol.json'),
              'continuation_manifest_sha256': digest(OUT / 'frozen-inputs.json')}
    for path in (ART / 'frozen-inputs.json', recovery / 'matrix-frozen-inputs.json', OUT / 'frozen-inputs.json'):
        manifest = read(path)
        entries = manifest.get('files', manifest)
        require(entries, 'empty frozen manifest')
        for name, record in entries.items():
            require(Path(name).stat().st_size == record['bytes'] and digest(name) == record['sha256'], 'frozen input changed: ' + name)
    client_hash = digest(ART / 'source-standalone/src/drama3/benchmark.py')
    labels = {f'c{c}-r{r}' for c in CONCURRENCY for r in range(3)}
    rows, provenance, audits, diagnostics = [], {}, {}, []
    for backend in BACKENDS:
        service_root = recovery / 'matrix' / backend if backend != 'sglang' else OUT / 'sglang'
        result_root = OUT / 'vllm-quality' if backend == 'vllm' else service_root
        result = read(result_root / 'result.json')
        expected_status = 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY' if backend == 'vllm' else 'COMPLETE_VALID_SERVICE_AND_QUALITY'
        require(result['status'] == expected_status, backend + ' incomplete')
        require(result['attempt_id'] == (RECOVERY if backend == 'standalone' else REPORT_ID), backend + ' attempt differs')
        for key, value in hashes.items():
            if backend == 'standalone' and key.startswith('continuation_'):
                continue
            require(result[key] == value, backend + ' protocol identity differs: ' + key)
        launch = read(service_root / 'service/launch.json')
        require(launch['source_sha'] == protocol['source_shas'][backend] and not launch['source_status']
                and launch['benchmark_client_sha256'] == client_hash, backend + ' source/client differs')
        wanted = labels - {'c64-r2'} if backend == 'vllm' else labels
        performance, quality = result['service'], result['quality']
        for name, audit in [('performance', performance), ('quality', quality)]:
            expected_audit_status = 'PASS_COMPLETED_GROUPS_ONLY' if backend == 'vllm' and name == 'performance' else 'PASS'
            require(audit['status'] == expected_audit_status and audit['groups_checked'] == len(wanted)
                    and audit['requests_checked'] == 1088 * len(wanted), backend + ' incomplete ' + name)
            require(len(audit['groups']) == len(wanted) and {g['label'] for g in audit['groups']} == wanted,
                    backend + ' missing/duplicate/unexpected ' + name + ' labels')
            require(all(g['requests'] == 1088 for g in audit['groups']), backend + ' short group')
        if backend == 'vllm':
            require(performance['planned_groups'] == 21 and performance['interrupted_group'] == 'c64-r2'
                    and performance['full_matrix_valid'] is False, 'partial vLLM audit lost failure scope')
        require(performance['all_wav_pcm_hashes_verified'] and quality['unfiltered_wer']
                and quality['all_sim_records_present'], backend + ' incomplete audio/quality coverage')
        require(performance['corpus_sha256'] == protocol['corpus_sha256']
                and performance['summary_sha256'] == digest(service_root / 'service/benchmark/summary.json')
                and quality['quality_sha256'] == digest(result_root / 'quality.json'), backend + ' raw audit drift')
        for group in performance['groups']:
            path = service_root / 'service/benchmark' / (group['label'] + '.jsonl')
            require(group['jsonl_sha256'] == digest(path), backend + ' raw group changed')
            require(all(math.isfinite(group[k]) for k in PERF), backend + ' nonfinite performance')
        for group in quality['groups']:
            require(all(math.isfinite(group[k]) for k in ('wer_percent', 'sim_mean')), backend + ' nonfinite quality')
        service_runner = read(service_root / 'service-runner/result.json')
        quality_runner = read(result_root / 'quality-runner/result.json')
        require(service_runner['status'] == ('FAIL' if backend == 'vllm' else 'PASS')
                and service_runner['post_gpu'] == [], backend + ' service cleanup incomplete')
        require(quality_runner['status'] == 'PASS' and quality_runner['post_gpu'] == [], backend + ' quality cleanup incomplete')
        if backend == 'vllm':
            failed = read(service_root / 'result.json')
            require(failed['status'] == 'STOPPED_FAILURE' and service_runner.get('error'), 'vLLM failure not preserved')
            require(result['failed_service_result_sha256'] == digest(service_root / 'result.json')
                    and result['failed_runner_result_sha256'] == digest(service_root / 'service-runner/result.json'), 'vLLM failure identity differs')
        pg = {g['label']: g for g in performance['groups']}
        qg = {g['label']: g for g in quality['groups']}
        audits[backend] = qg
        provenance[backend] = {'source_sha': launch['source_sha'], 'launch': launch,
                               'service_directory': str(service_root.relative_to(ART)),
                               'result_directory': str(result_root.relative_to(ART)),
                               'result_sha256': digest(result_root / 'result.json'),
                               'performance_audit': performance, 'quality_audit': quality}
        for c in CONCURRENCY:
            selected = [f'c{c}-r{r}' for r in range(3)]
            if backend == 'vllm' and c == 64:
                diagnostics.extend({'framework': backend, 'concurrency': c, 'label': label,
                                    'performance': pg[label], 'quality': qg[label], 'formal_three_run_row': False}
                                   for label in selected[:2])
                continue
            metrics = {key: stats([pg[label][key] for label in selected]) for key in PERF}
            metrics.update({key: stats([qg[label][key] for label in selected]) for key in ('wer_percent', 'sim_mean')})
            rows.append({'framework': backend, 'concurrency': c, 'repeats': 3, 'requests_per_repeat': 1088, 'metrics': metrics})
    for row in rows:
        backend, c = row['framework'], row['concurrency']
        if c == 64:
            row['quality_vs_vllm'] = {'available': False, 'reason': 'vLLM C64 lacks its third completed run; no three-run baseline or all-repeat quality verdict'}
            continue
        baseline = next(x for x in rows if x['framework'] == 'vllm' and x['concurrency'] == c)
        dw = row['metrics']['wer_percent']['median'] - baseline['metrics']['wer_percent']['median']
        ds = row['metrics']['sim_mean']['median'] - baseline['metrics']['sim_mean']['median']
        paired = [{'repeat': r, 'wer_delta_pp': audits[backend][f'c{c}-r{r}']['wer_percent'] - audits['vllm'][f'c{c}-r{r}']['wer_percent'],
                   'sim_delta': audits[backend][f'c{c}-r{r}']['sim_mean'] - audits['vllm'][f'c{c}-r{r}']['sim_mean']} for r in range(3)]
        row['quality_vs_vllm'] = {'available': True, 'wer_median_delta_pp': dw, 'sim_median_delta': ds,
                                  'median_within_reporting_bounds': dw <= 1 and ds >= -2,
                                  'paired_repeat_differences': paired,
                                  'all_repeats_within_reporting_bounds': all(x['wer_delta_pp'] <= 1 and x['sim_delta'] >= -2 for x in paired)}
    require(len(rows) == 20 and len(diagnostics) == 2, 'wrong formal/diagnostic coverage')
    preparation = read(ART / 'preparation.json')
    failed_root = recovery / 'matrix/vllm'
    return {'status': STATUS, 'optimization_acceptance_evaluated': False, 'complete_matrix': False,
            'report_id': REPORT_ID, 'selected_recovery': RECOVERY,
            'aggregation': 'Independent medians of three complete run metrics only; min/max are observed ranges. Repeat labels do not imply identical RNG/output lengths.',
            'excluded_from_metrics': ['original matrix stopped after monitoring timeout', 'all feasibility groups', 'vLLM C64 incomplete three-run row'],
            'expected_performance_groups': 63, 'performance_groups': 62, 'quality_groups': 62,
            'complete_group_requests': 67456, 'formal_rows': 20, 'rows': rows,
            'incomplete_rows': [{'framework': 'vllm', 'concurrency': 64, 'expected_repeats': 3,
                                 'completed_repeats': 2, 'failed_label': 'c64-r2', 'metrics': None,
                                 'reason': 'Service stalled; controller stopped after observed request-timeout cancellation evidence. Root cause unresolved.'}],
            'diagnostic_groups': diagnostics,
            'failed_group': {'framework': 'vllm', 'label': 'c64-r2', 'complete': False,
                             'service_result': str((failed_root / 'result.json').relative_to(ART)),
                             'service_result_sha256': digest(failed_root / 'result.json'),
                             'runner_result_sha256': digest(failed_root / 'service-runner/result.json'),
                             'request_failure_count': None, 'request_failure_count_note': 'Do not interpret duplicated server cancellation logs as unique failed requests.'},
            'quality_reporting_bounds': {'wer_increase_pp': 1, 'sim_decrease': 2, 'filters_results': False, 'available_concurrency': [1, 2, 4, 8, 16, 32]},
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
