"""Synthetic CPU tests for explicit incomplete-row reporting; no archiving or GPU."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summary = load('summarize_matrix')
archive = load('archive_comparison')
checks = []


def check(name, okay):
    if not okay:
        raise AssertionError(name)
    checks.append({'name': name, 'passed': True})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bad(name, call):
    try:
        call()
    except (RuntimeError, FileNotFoundError, KeyError):
        check(name, True)
    else:
        raise AssertionError('accepted: ' + name)


with tempfile.TemporaryDirectory(prefix='drama3-incomplete-cpu-') as tmp:
    art = Path(tmp) / 'campaign'
    out, recovery = art / summary.REPORT_ID, art / summary.RECOVERY
    out.mkdir(parents=True)
    recovery.mkdir()
    summary.OUT, summary.ART = out, art
    source = art / 'source-standalone/src/drama3/benchmark.py'
    source.parent.mkdir(parents=True)
    source.write_text('# frozen synthetic client\n')
    record = {'bytes': source.stat().st_size, 'sha256': summary.digest(source)}
    protocol = {'concurrency': list(summary.CONCURRENCY), 'repeats': 3, 'requests': 1088,
                'frameworks': list(summary.BACKENDS), 'source_shas': {b: b + '-sha' for b in summary.BACKENDS},
                'corpus_sha256': 'synthetic-corpus'}
    write(art / 'protocol.json', protocol)
    write(art / 'frozen-inputs.json', {'files': {str(source): record}})
    write(art / 'preparation.json', {'imports': {b: {'packages': {'torch': 'synthetic'}} for b in summary.BACKENDS}})
    write(recovery / 'matrix-protocol.json', {'attempt_id': summary.RECOVERY})
    write(recovery / 'matrix-frozen-inputs.json', {str(source): record})
    write(out / 'protocol.json', {'attempt_id': summary.REPORT_ID})
    write(out / 'frozen-inputs.json', {str(source): record})
    hashes = {'protocol_sha256': summary.digest(art / 'protocol.json'),
              'manifest_sha256': summary.digest(art / 'frozen-inputs.json'),
              'matrix_protocol_sha256': summary.digest(recovery / 'matrix-protocol.json'),
              'matrix_manifest_sha256': summary.digest(recovery / 'matrix-frozen-inputs.json'),
              'continuation_protocol_sha256': summary.digest(out / 'protocol.json'),
              'continuation_manifest_sha256': summary.digest(out / 'frozen-inputs.json')}
    for backend in summary.BACKENDS:
        service = out / 'sglang' if backend == 'sglang' else recovery / 'matrix' / backend
        result_root = out / 'vllm-quality' if backend == 'vllm' else service
        write(service / 'service/launch.json', {'source_sha': protocol['source_shas'][backend], 'source_status': '',
              'benchmark_client_sha256': summary.digest(source)})
        write(service / 'service/benchmark/summary.json', {'synthetic': True})
        write(result_root / 'quality.json', {'synthetic': True})
        pg, qg = [], []
        for c in summary.CONCURRENCY:
            for repeat in range(3):
                if backend == 'vllm' and c == 64 and repeat == 2:
                    continue
                label = f'c{c}-r{repeat}'
                raw = service / 'service/benchmark' / (label + '.jsonl')
                raw.write_text('{"synthetic": true}\n')
                pg.append({'label': label, 'requests': 1088, 'jsonl_sha256': summary.digest(raw),
                           **{k: 100 + c + repeat for k in summary.PERF}})
                qg.append({'label': label, 'requests': 1088, 'wer_percent': 20 + repeat if backend == 'standalone' else 1 + repeat,
                           'sim_mean': 60 + repeat})
        count = len(pg)
        perf = {'status': 'PASS_COMPLETED_GROUPS_ONLY' if backend == 'vllm' else 'PASS', 'groups_checked': count,
                'requests_checked': count * 1088, 'groups': pg, 'all_wav_pcm_hashes_verified': True,
                'corpus_sha256': protocol['corpus_sha256'], 'summary_sha256': summary.digest(service / 'service/benchmark/summary.json')}
        if backend == 'vllm':
            perf.update(planned_groups=21, interrupted_group='c64-r2', full_matrix_valid=False)
        quality = {'status': 'PASS', 'groups_checked': count, 'requests_checked': count * 1088, 'groups': qg,
                   'unfiltered_wer': True, 'all_sim_records_present': True, 'quality_sha256': summary.digest(result_root / 'quality.json')}
        write(service / 'service-runner/result.json', {'status': 'FAIL' if backend == 'vllm' else 'PASS',
              'post_gpu': [], 'error': 'KeyboardInterrupt' if backend == 'vllm' else None})
        write(result_root / 'quality-runner/result.json', {'status': 'PASS', 'post_gpu': []})
        result = {'status': 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY' if backend == 'vllm' else 'COMPLETE_VALID_SERVICE_AND_QUALITY',
                  'attempt_id': summary.RECOVERY if backend == 'standalone' else summary.REPORT_ID,
                  **hashes, 'service': perf, 'quality': quality}
        if backend == 'vllm':
            write(service / 'result.json', {'status': 'STOPPED_FAILURE', 'error': 'synthetic interrupted service'})
            result.update(failed_service_result_sha256=summary.digest(service / 'result.json'),
                          failed_runner_result_sha256=summary.digest(service / 'service-runner/result.json'))
        write(result_root / 'result.json', result)
    write(art / 'matrix/standalone/result.json', {'status': 'STOPPED_FAILURE', 'complete_groups': 7, 'fps': 999999})
    write(art / 'feasibility/vllm/result.json', {'fps': 888888})
    write(art / 'cpu-initial-probe.json', {'initial': True})
    write(art / 'failed-cache-attempt/result.json', {'failed': True})
    (art / 'failed-cache-attempt/server.log').write_text('failure retained\n')
    for relative in ('audio/test.wav', 'checkpoint/model.bin', 'cache/result.json', 'tensor.pt.gz'):
        path = art / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'excluded')
    aggregate = lambda: summary.aggregate(out)
    data = aggregate()
    check('62_complete_groups_67456_requests_not63', data['performance_groups'] == data['quality_groups'] == 62 and data['complete_group_requests'] == 67456 and data['complete_matrix'] is False)
    check('20_formal_rows_one_incomplete_row_two_diagnostics', len(data['rows']) == 20 and len(data['incomplete_rows']) == 1 and len(data['diagnostic_groups']) == 2)
    check('vllm_c64_no_formal_metric_or_median', not any(r['framework'] == 'vllm' and r['concurrency'] == 64 for r in data['rows']) and data['incomplete_rows'][0]['metrics'] is None)
    check('both_diagnostic_c64_runs_retained_unaggregated', [d['label'] for d in data['diagnostic_groups']] == ['c64-r0', 'c64-r1'])
    check('c64_quality_comparison_unavailable', all(not r['quality_vs_vllm']['available'] for r in data['rows'] if r['concurrency'] == 64))
    check('other_concurrencies_have_three_run_median_and_range', data['rows'][0]['metrics']['throughput_frames_s'] == {'median': 102, 'min': 101, 'max': 103})
    check('bad_quality_retained_and_no_optimization_acceptance', not data['rows'][0]['quality_vs_vllm']['median_within_reporting_bounds'] and data['optimization_acceptance_evaluated'] is False)
    check('failed_unique_request_count_not_invented', data['failed_group']['request_failure_count'] is None)
    check('aggregation_does_not_write', not (out / 'matrix-summary.json').exists())
    bad('other_report_directory_refused', lambda: summary.aggregate(recovery))
    result_path = out / 'vllm-quality/result.json'
    original = read_original = json.loads(result_path.read_text())
    mutations = {
        'missing_quality_refused': lambda x: x.update(status='RUNNING'),
        'wrong_attempt_refused': lambda x: x.update(attempt_id=summary.RECOVERY),
        'wrong_continuation_hash_refused': lambda x: x.update(continuation_protocol_sha256='wrong'),
        'vllm_service_not_relabelled_fullpass': lambda x: x['service'].update(status='PASS'),
        'vllm_failure_scope_required': lambda x: x['service'].update(full_matrix_valid=True),
        'arbitrary_missing_group_refused': lambda x: x['service']['groups'].pop(0),
        'duplicate_quality_label_refused': lambda x: x['quality']['groups'].__setitem__(1, x['quality']['groups'][0]),
        'fabricated_c64_r2_refused': lambda x: x['service']['groups'].append(dict(x['service']['groups'][-1], label='c64-r2')),
        'nonfinite_diagnostic_refused': lambda x: x['service']['groups'][-1].update(throughput_frames_s=float('nan')),
        'changed_failure_hash_refused': lambda x: x.update(failed_service_result_sha256='wrong'),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        write(result_path, changed)
        bad(name, aggregate)
        write(result_path, original)
    raw = recovery / 'matrix/vllm/service/benchmark/c64-r1.jsonl'
    saved = raw.read_text();raw.write_text('changed\n')
    bad('changed_diagnostic_raw_refused', aggregate)
    raw.write_text(saved)
    cleanup = out / 'sglang/quality-runner/result.json'
    write(cleanup, {'status': 'PASS', 'post_gpu': [{'pid': 1}]})
    bad('unclean_final_stage_refused', aggregate)
    write(cleanup, {'status': 'PASS', 'post_gpu': []})
    write(out / 'matrix-summary.json', aggregate())
    destination = Path(tmp) / 'archive'
    selection = archive.plan(art, destination, out)
    files = {r['source']: r for r in selection['files']}
    check('archive_plan_no_writes', not destination.exists())
    check('both_failures_pilots_and_initial_probe_preserved', all(p in files for p in ('matrix/standalone/result.json','recovery-attempt-01/matrix/vllm/result.json','feasibility/vllm/result.json','cpu-initial-probe.json','failed-cache-attempt/server.log')))
    check('raw_quality_and_logs_gzip', all(r['encoding'] == 'gzip' for p,r in files.items() if p.endswith(('.jsonl','.log','quality.json'))))
    check('worktrees_model_audio_caches_excluded', not any(p.startswith(('source-', 'audio/', 'checkpoint/', 'cache/')) or p.endswith('.pt.gz') for p in files))
    summary_path = out / 'matrix-summary.json'
    current = json.loads(summary_path.read_text())
    write(summary_path, dict(current, performance_groups=63))
    bad('archive_false_63group_summary_refused', lambda: archive.plan(art,destination,out))
    write(summary_path, current)
    destination.mkdir()
    bad('archive_existing_destination_refused', lambda: archive.plan(art,destination,out))
    check('archive_never_executed', list(destination.iterdir()) == [])

report = {'status': 'CPU_SYNTHETIC_INTERRUPTED_REPORT_CHECK_PASS',
          'scope': 'Synthetic fixtures only; no measured GPU result inferred. archive() never called; no GPU/repository writes.',
          'checks': checks, 'sources': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
          for name in ('summarize_matrix.py','archive_comparison.py','check_helpers.py')}}
(HERE/'helper-cpu-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks)}))
