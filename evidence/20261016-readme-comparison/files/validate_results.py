"""Read-only full-record checks for the fixed three-framework comparison."""
import hashlib
import json
import math
import statistics
import wave
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, minimum=None):
    require(isinstance(value, (int, float)) and math.isfinite(value), f'{name}: not finite')
    require(minimum is None or value >= minimum, f'{name}: below minimum')
    return value


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f'{name}: recorded {actual} != recomputed {expected}')


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def read_corpus(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    require(len(rows) == 1088, 'expected complete 1088 corpus')
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate corpus ID')
    return rows


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_identity(rows, samples, label, count):
    require(len(rows) == count, label + ': wrong record count')
    expected = {f'{label}-{i:05d}': samples[i % len(samples)]['id'] for i in range(count)}
    require(len({r['request_id'] for r in rows}) == count, label + ': duplicate request')
    require({r['request_id']: r['sample_id'] for r in rows} == expected,
            label + ': missing or mismatched sample/request IDs')
    require(all(r['success'] and r['http_status'] == 200 for r in rows), label + ': failed request')


def validate_service(output_dir, corpus, concurrency, repeats):
    output_dir = Path(output_dir)
    bench = output_dir / 'benchmark'
    samples = read_corpus(corpus)
    manifest = json.loads((bench / 'manifest.json').read_text())
    require(manifest['corpus_sha256'] == digest(corpus), 'benchmark corpus differs')
    require(manifest['requests'] == 1088 and manifest['repeats'] == repeats
            and manifest['concurrency'] == list(concurrency), 'benchmark shape differs')
    require(manifest['max_frames'] == 500 and manifest['warmup'] == 8
            and manifest['save_audio'] and not manifest['warm_references'], 'benchmark controls differ')
    require(manifest['connection_policy'] == 'fresh TCP connection per request; no automatic retries',
            'connection policy differs')
    summaries = json.loads((bench / 'summary.json').read_text())
    require(len(summaries) == len(concurrency) * repeats, 'missing/extra summary groups')
    groups = {(r['concurrency'], r['repeat']): r for r in summaries}
    require(len(groups) == len(summaries), 'duplicate summary group')
    checked = []
    for c in concurrency:
        label = f'c{c}-warmup'
        warmup = json.loads((bench / f'{label}.json').read_text())
        check_identity(warmup, samples, label, max(8, c))
        for repeat in range(repeats):
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
    return {'status': 'PASS', 'groups_checked': len(checked), 'requests_checked': 1088 * len(checked),
            'all_wav_pcm_hashes_verified': True, 'corpus_sha256': digest(corpus),
            'summary_sha256': digest(bench / 'summary.json'), 'groups': checked}


def validate_quality(path, service_dir, corpus, labels):
    samples = {r['id']: r for r in read_corpus(corpus)}
    data = json.loads(Path(path).read_text())
    require(set(data['groups']) == set(labels), 'missing/extra quality groups')
    checked = []
    for label in labels:
        group = data['groups'][label]
        rows = group['records']
        service = read_records(Path(service_dir) / 'benchmark' / f'{label}.jsonl')
        expected = {r['request_id']: r['sample_id'] for r in service}
        require(len(rows) == len({r['request_id'] for r in rows}) == 1088, label + ': quality coverage')
        require({r['request_id']: r['sample_id'] for r in rows} == expected,
                label + ': quality IDs differ')
        errors = words = 0
        for row in rows:
            require(row['is_success'], label + ': quality failure')
            require(row['target_text'] == samples[row['sample_id']]['text'], label + ': wrong target text')
            finite(row['wer'], label + ' sample WER', 0)
            finite(row['speaker_similarity'], label + ' sample SIM')
            for key in ('substitutions', 'deletions', 'insertions', 'hits'):
                require(isinstance(row[key], int) and row[key] >= 0, label + ': invalid edit counts')
            errors += row['substitutions'] + row['deletions'] + row['insertions']
            words += row['hits'] + row['substitutions'] + row['deletions']
        wer = group['wer']; sim = group['similarity']
        require(group['complete'] and wer['total_samples'] == wer['evaluated'] == 1088
                and sim['total'] == sim['evaluated'] == 1088
                and wer['skipped'] == sim['skipped'] == 0, label + ': incomplete scores')
        require(words > 0, label + ': no reference words')
        close(wer['wer_corpus'], errors / words, label + ' unfiltered corpus WER')
        close(sim['mean'], statistics.mean(r['speaker_similarity'] for r in rows), label + ' SIM mean')
        checked.append({'label': label, 'requests': 1088, 'wer_percent': wer['wer_corpus'] * 100,
                        'sim_mean': sim['mean'], 'errors': errors, 'reference_words': words})
    return {'status': 'PASS', 'groups_checked': len(checked), 'requests_checked': len(checked) * 1088,
            'unfiltered_wer': True, 'all_sim_records_present': True, 'quality_sha256': digest(path),
            'groups': checked}
