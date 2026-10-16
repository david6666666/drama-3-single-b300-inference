"""Synthetic CPU checks only; never execute archive(), launch servers, or use CUDA."""
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary, archive = load('summarize_matrix'), load('archive_comparison')
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


with tempfile.TemporaryDirectory(prefix='drama3-recovery-cpu-') as temporary:
    root = Path(temporary) / 'campaign'
    attempt = root / summary.ATTEMPT_ID
    attempt.mkdir(parents=True)
    summary.OUT, summary.CAMPAIGN = attempt, root
    source = root / 'source-standalone/src/drama3/benchmark.py'
    source.parent.mkdir(parents=True)
    source.write_text('# synthetic frozen client\n')
    record = lambda path: {'bytes': path.stat().st_size, 'sha256': summary.digest(path)}
    protocol = {'concurrency': list(summary.CONCURRENCY), 'repeats': 3, 'requests': 1088,
                'frameworks': list(summary.BACKENDS), 'source_shas': {b: b + '-sha' for b in summary.BACKENDS},
                'corpus_sha256': 'synthetic-corpus'}
    write(root / 'protocol.json', protocol)
    write(root / 'frozen-inputs.json', {'files': {str(source): record(source)}})
    write(root / 'preparation.json', {'imports': {b: {'packages': {'torch': 'synthetic'}} for b in summary.BACKENDS}})
    write(attempt / 'matrix-protocol.json', {'attempt_id': summary.ATTEMPT_ID,
          'base_protocol_sha256': summary.digest(root / 'protocol.json'),
          'base_manifest_sha256': summary.digest(root / 'frozen-inputs.json')})
    write(attempt / 'matrix-frozen-inputs.json', {str(source): record(source)})
    labels = [f'c{c}-r{r}' for c in summary.CONCURRENCY for r in range(3)]
    for b in summary.BACKENDS:
        directory = attempt / 'matrix' / b
        write(directory / 'service/launch.json', {'source_sha': protocol['source_shas'][b], 'source_status': '',
              'benchmark_client_sha256': summary.digest(source)})
        write(directory / 'service/benchmark/summary.json', {'synthetic': True})
        write(directory / 'quality.json', {'synthetic': True})
        pg, qg = [], []
        for c in summary.CONCURRENCY:
            for r in range(3):
                label = f'c{c}-r{r}'
                path = directory / 'service/benchmark' / (label + '.jsonl')
                path.write_text('{"synthetic": true}\n')
                pg.append({'label': label, 'requests': 1088, 'jsonl_sha256': summary.digest(path),
                           **{k: 100 + c + r for k in summary.PERF}})
                qg.append({'label': label, 'requests': 1088,
                           'wer_percent': 20 + r if b == 'standalone' else 1 + r, 'sim_mean': 60 + r})
        perf = {'status': 'PASS', 'groups_checked': 21, 'requests_checked': 22848, 'groups': pg,
                'all_wav_pcm_hashes_verified': True, 'corpus_sha256': protocol['corpus_sha256'],
                'summary_sha256': summary.digest(directory / 'service/benchmark/summary.json')}
        quality = {'status': 'PASS', 'groups_checked': 21, 'requests_checked': 22848, 'groups': qg,
                   'unfiltered_wer': True, 'all_sim_records_present': True,
                   'quality_sha256': summary.digest(directory / 'quality.json')}
        result = {'status': 'COMPLETE_VALID_SERVICE_AND_QUALITY', 'attempt_id': summary.ATTEMPT_ID,
                  'protocol_sha256': summary.digest(root / 'protocol.json'),
                  'manifest_sha256': summary.digest(root / 'frozen-inputs.json'),
                  'matrix_protocol_sha256': summary.digest(attempt / 'matrix-protocol.json'),
                  'matrix_manifest_sha256': summary.digest(attempt / 'matrix-frozen-inputs.json'),
                  'service': perf, 'quality': quality}
        write(directory / 'result.json', result)
        for phase in ('service-runner', 'quality-runner'):
            write(directory / phase / 'result.json', {'status': 'PASS', 'post_gpu': []})
    # Older attempt and pilots deliberately conflict; they may only be archived.
    write(root / 'matrix/standalone/result.json', {'status': 'STOPPED_FAILURE', 'completed_groups': 7})
    write(root / 'matrix/standalone/service/benchmark/summary.json', {'throughput_frames_s': 999999})
    (root / 'matrix/standalone/service/benchmark/c1-r0.jsonl').write_text('{"failed_attempt": true}\n')
    write(root / 'feasibility/standalone/result.json', {'throughput_frames_s': 888888})
    write(root / 'failed-cache-attempt/result.json', {'failed': True})
    write(root / 'cpu-initial-probe.json', {'initial': True})
    (root / 'failed-cache-attempt/server.log').write_text('failure retained\n')
    for rel in ('audio/test.wav', 'checkpoints/model.bin', 'cache/result.json', 'tensor.pt.gz'):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'excluded')
    aggregate = lambda: summary.aggregate(root / 'protocol.json', root / 'frozen-inputs.json', attempt)
    output = aggregate()
    check('complete_selected_attempt_63_performance_63_quality', output['performance_groups'] == output['quality_groups'] == 63)
    row = output['rows'][0]
    check('only_recovery_metrics_three_run_median_range', row['metrics']['throughput_frames_s'] == {'median': 102, 'min': 101, 'max': 103})
    check('failed_attempt_and_pilots_explicitly_excluded', output['selected_matrix'] == 'recovery-attempt-01/matrix' and len(output['excluded_from_metrics']) == 3)
    check('bad_quality_reported_without_filter_or_acceptance', not row['quality_vs_vllm']['median_within_reporting_bounds'] and output['optimization_acceptance_evaluated'] is False)
    check('aggregate_does_not_write_summary', not (attempt / 'matrix-summary.json').exists())
    bad('original_attempt_cannot_be_selected', lambda: summary.aggregate(root / 'protocol.json', root / 'frozen-inputs.json', root))
    bad('other_recovery_cannot_be_selected', lambda: summary.aggregate(root / 'protocol.json', root / 'frozen-inputs.json', root / 'recovery-attempt-02'))
    bad('wrong_base_paths_refused', lambda: summary.aggregate(attempt / 'protocol.json', root / 'frozen-inputs.json', attempt))
    result_path = attempt / 'matrix/standalone/result.json'
    original = json.loads(result_path.read_text())
    mutations = {
        'incomplete_recovery_not_filled_from_prior': lambda x: x.update(status='STOPPED_FAILURE'),
        'foreign_attempt_result_rejected': lambda x: x.update(attempt_id='original'),
        'wrong_matrix_protocol_rejected': lambda x: x.update(matrix_protocol_sha256='wrong'),
        'missing_run_rejected': lambda x: x['service']['groups'].pop(),
        'duplicate_run_rejected': lambda x: x['quality']['groups'].__setitem__(1, x['quality']['groups'][0]),
        'nonfinite_metric_rejected': lambda x: x['service']['groups'][0].update(throughput_frames_s=float('nan')),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        write(result_path, changed)
        bad(name, aggregate)
        write(result_path, original)
    raw = attempt / 'matrix/standalone/service/benchmark/c1-r0.jsonl'
    raw_original = raw.read_text()
    raw.write_text('changed\n')
    bad('changed_raw_jsonl_hash_rejected', aggregate)
    raw.write_text(raw_original)
    write(attempt / 'matrix-summary.json', aggregate())
    destination = Path(temporary) / 'new-archive'
    planned = archive.plan(root, destination, attempt)
    files = {r['source']: r for r in planned['files']}
    check('archive_default_plan_has_no_writes', not destination.exists())
    check('all_attempts_pilots_initial_probes_retained', all(p in files for p in (
        'matrix/standalone/result.json', 'matrix/standalone/service/benchmark/c1-r0.jsonl',
        'feasibility/standalone/result.json', 'failed-cache-attempt/result.json', 'cpu-initial-probe.json')))
    check('all_raw_quality_logs_gzip', all(r['encoding'] == 'gzip' for p, r in files.items() if p.endswith(('.jsonl', '.log', 'quality.json'))))
    check('worktrees_weights_audio_caches_excluded', not any(p.startswith(('source-', 'audio/', 'checkpoints/', 'cache/')) or p.endswith('.pt.gz') for p in files))
    check('selected_summary_plain_and_attempt_scoped', files['recovery-attempt-01/matrix-summary.json']['encoding'] == 'plain' and planned['selected_attempt'] == 'recovery-attempt-01')
    check('all_source_uncompressed_hashes_bytes_recorded', all(len(r['uncompressed_sha256']) == 64 and r['uncompressed_bytes'] >= 0 for r in files.values()))
    bad('archive_original_attempt_selection_refused', lambda: archive.plan(root, destination, root))
    cleanup = attempt / 'matrix/standalone/service-runner/result.json'
    write(cleanup, {'status': 'PASS', 'post_gpu': [{'pid': 1}]})
    bad('archive_incomplete_cleanup_refused', lambda: archive.plan(root, destination, attempt))
    write(cleanup, {'status': 'PASS', 'post_gpu': []})
    summary_path = attempt / 'matrix-summary.json'
    stored_summary = json.loads(summary_path.read_text())
    write(summary_path, dict(stored_summary, attempt_id='original'))
    bad('archive_wrong_summary_attempt_refused', lambda: archive.plan(root, destination, attempt))
    write(summary_path, dict(stored_summary, performance_groups=7))
    bad('archive_incomplete_summary_refused', lambda: archive.plan(root, destination, attempt))
    write(summary_path, stored_summary)
    destination.mkdir()
    bad('archive_existing_destination_refused', lambda: archive.plan(root, destination, attempt))
    check('archive_function_never_invoked', list(destination.iterdir()) == [])

report = {'status': 'CPU_SYNTHETIC_HELPER_CHECK_PASS',
          'scope': 'Synthetic fixtures only; not actual matrix validation. archive() never invoked; no GPU/server/repository writes.',
          'checks': checks, 'helpers': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
          for name in ('summarize_matrix.py', 'archive_comparison.py', 'check_helpers.py')}}
(HERE / 'helper-cpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'checks': len(checks)}))
