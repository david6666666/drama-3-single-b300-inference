"""Archive an audited incomplete-row comparison; default is read-only, no GPU work."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

from summarize_matrix import aggregate

ATTEMPT = Path(__file__).resolve().parent
ART = ATTEMPT.parent
ATTEMPT_ID = "remaining-sglang-report-01"
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
    require(attempt.resolve() == root.resolve() / ATTEMPT_ID, 'wrong explicit report directory')
    summary = read(attempt / 'matrix-summary.json')
    recomputed = aggregate(attempt)
    require(summary == recomputed, 'stored summary differs from complete current partial-report audits')
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
            'metrics_scope': '60 complete groups and one invalid partial group;19 formal three-run rows;two incomplete rows;all failures preserved',
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

[The descriptive summary](SUMMARY_PATH) retains two incomplete concurrency rows;
this is not a complete 63-group matrix or an optimization acceptance PASS.
There are 60 complete performance/quality groups (65,280 requests), 19 formal
three-run rows, and 1,087 scored audio outputs from one invalid partial group.
The latter group retains all 1,088 request records including its one HTTP500
failure. Total scored saved audio: 66,367. Partial-output quality is not a score
for the full 1,088-request workload.

vLLM C64 has two complete diagnostic runs; its third run stalled and was stopped
after request-timeout cancellation evidence. SGLang C2 has one complete diagnostic
run and a second invalid run with 1,087 successes and one HTTP500/no-audio response.
Its C2 third run was not attempted. No formal three-run median is reported for
either incomplete row. The invalid group's recorded performance metrics appear
only as failure diagnostics. Duplicated cancellation logs are not unique request counts.

Selected standalone and vLLM records remain under `files/recovery-attempt-01/matrix/`.
The first SGLang server records remain under `files/interrupted-matrix-report-01/sglang/`;
its five groups are scored under `files/remaining-sglang-report-01/existing-quality/`.
Only untouched C4/C8/C16/C32/C64 were measured in the new SGLang server session,
under `files/remaining-sglang-report-01/remaining/`. This new startup/capture/RNG
boundary is explicit; SGLang did not complete all21 groups in one server lifecycle.
The 20 saved vLLM groups are scored under `files/interrupted-matrix-report-01/vllm-quality/`.

All three stops remain preserved: the original monitor timeout, the vLLM service
stall, and the SGLang no-audio response. `files/matrix/` retains the original
monitor-failed attempt; its seven completed groups are excluded from these results.
`files/feasibility/` retains separate pilots, also excluded. No failed row is retried,
replaced or reconstructed. Missing interrupted-group records are not invented.

Raw request JSONL, quality JSON and logs use gzip; small protocols/configurations,
commands, helpers and reviews remain plain. [The index](archive-index.json) records
exact original/uncompressed SHA-256 and byte counts plus stored-file checksums.
Existing gzip bytes are preserved; decompress gzip to recover exact source bytes.

WAV/PCM audio stays on the local experiment machine and is deliberately excluded.
Raw records retain PCM hashes, sample counts and timing provenance; quality records
retain successful scores and the original failed record. Worktrees, Git directories,
model weights, checkpoints and caches are excluded. Local WAV files are required
to rerun scoring; this archive does not contain them.
'''
    (destination / 'README.md').write_text(readme.replace('SUMMARY_PATH', summary_path))
    return len(selection['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.attempt.resolve() == ATTEMPT, "report directory must match this helper")
    selection = plan(ART, args.destination, args.attempt)
    if args.execute:
        print(json.dumps({'status': 'COMPLETE_ARCHIVE', 'files': archive(selection), 'destination': str(args.destination)}))
    else:
        print(json.dumps({'status': 'DRY_RUN_NO_WRITES', 'files': len(selection['files']),
                          'destination': str(args.destination), 'excluded': selection['excluded']}, indent=2))
