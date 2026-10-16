"""Archive a completed comparison; default is a read-only plan, never a GPU launch."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parent
ART = ATTEMPT.parent
ATTEMPT_ID = "recovery-attempt-01"
DESTINATION = Path('/home/zjy/code/hsliu/drama-3-single-b300-inference/evidence/20261016-readme-comparison')
BACKENDS = ('standalone', 'vllm', 'sglang')
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
SKIP_DIRS = {'__pycache__', 'models', 'model', 'checkpoints', 'checkpoint', 'caches', 'cache', 'venv', 'venvs', 'worktrees', 'hf_modules', 'hf-modules', 'jit-cache', 'triton-cache', 'torchinductor-cache'}
SKIP_SUFFIXES = {'.wav', '.pcm', '.mp3', '.flac', '.ogg', '.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.pyc'}
TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh', '.yaml', '.yml', '.toml', '.csv', '.patch', '.diff', '.lock', '.conf', '.svg', '.gz'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path, *, uncompressed=False):
    digest, count = hashlib.sha256(), 0
    opener = gzip.open if uncompressed and path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
    return {'sha256': digest.hexdigest(), 'bytes': count}


def require_complete(root, attempt):
    require(attempt.resolve() == root.resolve() / ATTEMPT_ID, "wrong explicit recovery attempt")
    summary = read(attempt / "matrix-summary.json")
    require(summary["attempt_id"] == ATTEMPT_ID
            and summary["selected_matrix"] == ATTEMPT_ID + "/matrix", "wrong summary attempt")
    require(summary['status'] == 'COMPLETE_DESCRIPTIVE_COMPARISON'
            and summary['performance_groups'] == summary['quality_groups'] == 63,
            'matrix summary is incomplete')
    require(summary['optimization_acceptance_evaluated'] is False, 'unexpected optimization acceptance claim')
    require(len(summary['rows']) == 21 and {(r['framework'], r['concurrency']) for r in summary['rows']}
            == {(b, c) for b in BACKENDS for c in CONCURRENCY}, 'missing matrix rows')
    require(all(r['repeats'] == 3 and r['requests_per_repeat'] == 1088 for r in summary['rows']), 'partial matrix rows')
    for key, name in [('protocol_sha256', 'protocol.json'), ('manifest_sha256', 'frozen-inputs.json'),
                      ('matrix_protocol_sha256', 'matrix-protocol.json'), ('matrix_manifest_sha256', 'matrix-frozen-inputs.json')]:
        base = attempt if key.startswith("matrix_") else root
        require(summary[key] == fingerprint(base / name)["sha256"], name + " changed")
    labels = {f'c{c}-r{r}' for c in CONCURRENCY for r in range(3)}
    for backend in BACKENDS:
        directory = attempt / 'matrix' / backend
        result = read(directory / 'result.json')
        provenance = summary['provenance'][backend]
        require(result['attempt_id'] == ATTEMPT_ID, backend + ' wrong result attempt')
        require(result['status'] == 'COMPLETE_VALID_SERVICE_AND_QUALITY'
                and provenance['result_sha256'] == fingerprint(directory / 'result.json')['sha256'], backend + ' incomplete or changed')
        for kind, summary_key in [('service', 'performance_audit'), ('quality', 'quality_audit')]:
            audit = result[kind]
            require(audit == provenance[summary_key] and audit['status'] == 'PASS'
                    and audit['groups_checked'] == 21 and audit['requests_checked'] == 22848,
                    backend + ' incomplete ' + kind)
            require(len(audit['groups']) == 21 and {g['label'] for g in audit['groups']} == labels,
                    backend + ' missing/duplicate ' + kind + ' groups')
        for phase in ('service-runner', 'quality-runner'):
            cleanup = read(directory / phase / 'result.json')
            require(cleanup['status'] == 'PASS' and cleanup['post_gpu'] == [], backend + ' cleanup incomplete')
        require(result['service']['summary_sha256'] == fingerprint(directory / 'service/benchmark/summary.json')['sha256']
                and result['quality']['quality_sha256'] == fingerprint(directory / 'quality.json')['sha256'], backend + ' raw result changed')
        for group in result['service']['groups']:
            path = directory / 'service/benchmark' / (group['label'] + '.jsonl')
            require(group['jsonl_sha256'] == fingerprint(path)['sha256'], backend + ' raw records changed')
    return summary


def plan(root, destination, attempt):
    root, destination, attempt = root.resolve(), destination.resolve(), attempt.resolve()
    require(not destination.exists(), 'destination already exists; never overwrite')
    require(not destination.is_relative_to(root), 'destination cannot be inside artifact source')
    require_complete(root, attempt)
    records, archived_names, excluded = [], set(), {'directories': [], 'files': 0}
    for base, directories, names in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(directories):
            path = Path(base) / name
            skip = name.startswith(('.', 'source-')) or name in SKIP_DIRS or (path / '.git').exists()
            if skip:
                excluded['directories'].append(str(path.relative_to(root)))
            else:
                require(not path.is_symlink(), 'refuse symlink directory: ' + str(path))
                kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = Path(base) / name
            if name.startswith('.') or any(suffix.lower() in SKIP_SUFFIXES for suffix in path.suffixes):
                excluded['files'] += 1
                continue
            require(not path.is_symlink(), 'refuse symlink file: ' + str(path))
            require(path.suffix.lower() in TEXT_SUFFIXES or name in {'LICENSE', 'NOTICE'}, 'unclassified artifact: ' + str(path))
            raw = fingerprint(path)
            compressed = path.suffix == '.gz' or path.suffix in {'.log', '.jsonl'} or name == 'quality.json' or raw['bytes'] > 1024 * 1024
            relative = path.relative_to(root)
            stored = Path('files') / (str(relative) + ('.gz' if compressed and path.suffix != '.gz' else ''))
            require(str(stored) not in archived_names, 'archive filename collision: ' + str(stored))
            archived_names.add(str(stored))
            records.append({'source': str(relative), 'archive': str(stored), 'encoding': 'gzip' if compressed else 'plain',
                            'source_sha256': raw['sha256'], 'source_bytes': raw['bytes'],
                            **{'uncompressed_' + k: v for k, v in fingerprint(path, uncompressed=True).items()}})
    return {'source': str(root), 'destination': str(destination), 'selected_attempt': str(attempt.relative_to(root)),
            'metrics_scope': ATTEMPT_ID + '/matrix only; original attempt and feasibility are evidence only',
            'files': records, 'excluded': excluded}


def archive(selection):
    root, destination = Path(selection['source']), Path(selection['destination'])
    attempt = root / selection['selected_attempt']
    destination.mkdir(parents=False, exist_ok=False)
    for record in selection['files']:
        source, target = root / record['source'], destination / record['archive']
        target.parent.mkdir(parents=True, exist_ok=True)
        require(fingerprint(source) == {'sha256': record['source_sha256'], 'bytes': record['source_bytes']}, 'source changed before copy')
        if record['encoding'] == 'gzip' and source.suffix != '.gz':
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                with gzip.GzipFile(filename='', mode='wb', fileobj=outgoing, mtime=0) as compressor:
                    shutil.copyfileobj(incoming, compressor, length=1024 * 1024)
        else:
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        require(fingerprint(source) == {'sha256': record['source_sha256'], 'bytes': record['source_bytes']}, 'source changed during copy')
        require(fingerprint(target, uncompressed=True) == {'sha256': record['uncompressed_sha256'], 'bytes': record['uncompressed_bytes']}, 'archive round-trip differs')
        record.update({'stored_' + k: v for k, v in fingerprint(target).items()})
    require_complete(root, attempt)
    index = dict(selection, status='COMPLETE_ARCHIVE', archiver_sha256=fingerprint(Path(__file__))['sha256'])
    (destination / 'archive-index.json').write_text(json.dumps(index, indent=2) + '\n')
    summary_path = next(r['archive'] for r in selection['files'] if r['source'] == str(Path(selection['selected_attempt']) / 'matrix-summary.json'))
    readme = '''# Latest-source Drama 3 comparison archive

This is a descriptive comparison, not an optimization acceptance PASS. The complete
63 performance groups and 63 quality groups are summarized in
[the matrix summary](SUMMARY_PATH). Source revisions, native configuration
differences and measurement limits remain in the protocol and preparation records.

`files/recovery-attempt-01/matrix/` alone supplies the final 63 performance and
63 quality groups. `files/matrix/` preserves the original stopped attempt, including
its seven completed groups and interrupted run; none contributes to this summary.
`files/feasibility/` retains separate pilots, also excluded from matrix metrics.
All other recorded attempts, failures and initial CPU probes remain under their
original relative names. No failure is replaced with a later successful attempt.

Raw request JSONL, quality JSON and logs use gzip; small protocol/configuration,
command, helper and review files remain plain. [The index](archive-index.json) records
exact original and uncompressed SHA-256/byte counts plus stored-file checksums.
Decompress a gzip file to recover its exact source bytes. Pre-existing gzip files
retain their original compressed bytes, with separate decompressed checksums.

WAV/PCM audio remains on the local experiment machine and is deliberately excluded.
Request JSONL retains each PCM hash, sample count and timing provenance; quality
records retain every scored output. Worktrees, Git directories, model weights,
checkpoints and caches are excluded. This archive does not contain enough data to
re-score the audio without the retained local WAV files.
'''
    (destination / 'README.md').write_text(readme.replace('SUMMARY_PATH', summary_path))
    return len(selection['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.attempt.resolve() == ATTEMPT, "attempt must match this recovery helper")
    selection = plan(ART, args.destination, args.attempt)
    if args.execute:
        print(json.dumps({'status': 'COMPLETE_ARCHIVE', 'files': archive(selection), 'destination': str(args.destination)}))
    else:
        print(json.dumps({'status': 'DRY_RUN_NO_WRITES', 'files': len(selection['files']),
                          'destination': str(args.destination), 'excluded': selection['excluded']}, indent=2))
