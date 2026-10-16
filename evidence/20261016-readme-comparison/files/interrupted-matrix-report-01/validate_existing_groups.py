"""Read-only audit of completed outputs from the interrupted vLLM matrix.

The original 21-group benchmark manifest is required unchanged. This checker
permits only its exact 20-group prefix, never a changed plan or arbitrary subset.
The incomplete C64 third repetition is not an accepted three-run result.
Original full-record/WAV/timing checks are preserved; no files are synthesized.
"""
import hashlib
import json
import statistics
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate_results import (
    check_identity, close, digest, finite, percentile, read_corpus, read_records,
    require,
)

def validate_existing_groups(output_dir, corpus):
    """Validate exactly the 20 complete groups before the documented vLLM stall."""
    concurrency = [1, 2, 4, 8, 16, 32, 64]
    repeats = 3
    expected_groups = [(c, r) for c in concurrency for r in range(repeats)][:-1]
    output_dir = Path(output_dir)
    bench = output_dir / 'benchmark'
    samples = read_corpus(corpus)
    manifest = json.loads((bench / 'manifest.json').read_text())
    require(manifest['backend'] == 'vllm', 'partial validator is only for this vLLM interruption')
    require(manifest['corpus_sha256'] == digest(corpus), 'benchmark corpus differs')
    require(manifest['requests'] == 1088 and manifest['repeats'] == repeats
            and manifest['concurrency'] == list(concurrency), 'benchmark shape differs')
    require(manifest['max_frames'] == 500 and manifest['warmup'] == 8
            and manifest['save_audio'] and not manifest['warm_references'], 'benchmark controls differ')
    require(manifest['connection_policy'] == 'fresh TCP connection per request; no automatic retries',
            'connection policy differs')
    summaries = json.loads((bench / 'summary.json').read_text())
    require([(r['concurrency'], r['repeat']) for r in summaries] == expected_groups,
            'expected exactly the original 20-group prefix; no gaps, retries or reordering')
    require(not (bench / 'c64-r2.jsonl').exists(),
            'interrupted group now has records; must review rather than silently exclude it')
    groups = {(r['concurrency'], r['repeat']): r for r in summaries}
    require(len(groups) == len(summaries), 'duplicate summary group')
    checked = []
    for c in concurrency:
        label = f'c{c}-warmup'
        warmup = json.loads((bench / f'{label}.json').read_text())
        check_identity(warmup, samples, label, max(8, c))
        for repeat in range(repeats):
            if (c, repeat) not in expected_groups:
                continue
            label = f'c{c}-r{repeat}'
            records_path = bench / f'{label}.jsonl'
            rows = read_records(records_path)
            check_identity(rows, samples, label, 1088)
            summary = groups[(c, repeat)]
            require(summary['valid'] and summary['requests'] == summary['completed'] == 1088
                    and summary['failed'] == 0, label + ': invalid summary')
            elapsed = finite(summary['makespan_s'], label + ' makespan', 1e-12)
            frame_sum = 0.0
            for row in rows:
                rid = row['request_id']
                for key in ('dispatch_s', 'ttfa_s', 'last_audio_s', 'completion_s', 'rtf', 'rms', 'peak'):
                    finite(row[key], rid + ' ' + key, 0)
                require(row['ttfa_s'] <= row['last_audio_s'] <= row['completion_s'], rid + ': invalid times')
                arrivals = row['chunk_arrivals_s']; chunk_bytes = row['chunk_bytes']
                require(len(arrivals) == len(chunk_bytes) == row['transport_chunks'] > 0,
                        rid + ': invalid chunks')
                require(arrivals == sorted(arrivals) and all(x > 0 for x in chunk_bytes), rid + ': invalid chunk sequence')
                close(row['ttfa_s'], arrivals[0], rid + ' TTFA')
                close(row['last_audio_s'], arrivals[-1], rid + ' last audio')
                with wave.open(str(bench / f'{rid}.wav'), 'rb') as wav:
                    require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype())
                            == (44100, 1, 2, 'NONE'), rid + ': wrong WAV format')
                    require(wav.getnframes() == row['audio_samples'] > 0, rid + ': wrong WAV samples')
                    pcm = wav.readframes(wav.getnframes())
                require(sum(chunk_bytes) == len(pcm) == row['audio_samples'] * 2,
                        rid + ': incomplete PCM')
                require(hashlib.sha256(pcm).hexdigest() == row['audio_sha256'], rid + ': PCM hash differs')
                close(row['acoustic_frames'], row['audio_samples'] / 2048, rid + ' acoustic frames')
                close(row['audio_duration_s'], row['audio_samples'] / 44100, rid + ' duration')
                close(row['rtf'], row['last_audio_s'] / row['audio_duration_s'], rid + ' RTF')
                frame_sum += row['audio_samples'] / 2048
            observed_span = max(r['dispatch_s'] + r['completion_s'] for r in rows) - min(r['dispatch_s'] for r in rows)
            require(elapsed + 1e-7 >= observed_span, label + ': makespan excludes request tail')
            recomputed = {
                'throughput_frames_s': frame_sum / elapsed,
                'ttfa_p50_ms': percentile([r['ttfa_s'] for r in rows], .5) * 1000,
                'ttfa_p95_ms': percentile([r['ttfa_s'] for r in rows], .95) * 1000,
                'rtf_mean': statistics.mean(r['rtf'] for r in rows),
            }
            for key, value in recomputed.items():
                finite(summary[key], label + ' ' + key, 0)
                close(summary[key], value, label + ' ' + key)
            checked.append({'label': label, 'requests': len(rows), 'frames': frame_sum,
                            'makespan_s': elapsed, 'jsonl_sha256': digest(records_path), **recomputed})
    return {'status': 'PASS_COMPLETED_GROUPS_ONLY', 'planned_groups': 21,
            'interrupted_group': 'c64-r2', 'full_matrix_valid': False,
            'groups_checked': len(checked), 'requests_checked': 1088 * len(checked),
            'all_wav_pcm_hashes_verified': True, 'corpus_sha256': digest(corpus),
            'summary_sha256': digest(bench / 'summary.json'), 'groups': checked}

