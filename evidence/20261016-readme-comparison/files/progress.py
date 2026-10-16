"""Read-only concise progress; HTTP headers are not completed-run validation."""
import json
from datetime import datetime, timezone
from pathlib import Path

base = Path(__file__).resolve().parent
recovery = base / 'recovery-attempt-01/matrix'
continuation = base / 'interrupted-matrix-report-01'
remaining = base / 'remaining-sglang-report-01'
out = {'utc': datetime.now(timezone.utc).strftime('%H:%M:%S'), 'stages': []}
stages = [
    ('standalone', recovery / 'standalone', recovery / 'standalone/service'),
    ('vllm-service', recovery / 'vllm', recovery / 'vllm/service'),
    ('vllm-quality', continuation / 'vllm-quality', recovery / 'vllm/service'),
    ('sglang', continuation / 'sglang', continuation / 'sglang/service'),
    ('sglang-saved-quality', remaining / 'existing-quality', continuation / 'sglang/service'),
    ('sglang-remaining', remaining / 'remaining', remaining / 'remaining/service'),
]
for stage, directory, service in stages:
    result_path = directory / 'result.json'
    if not result_path.exists():
        continue
    try:
        result = json.loads(result_path.read_text())
        row = {'stage': stage, 'status': result['status']}
        if result['status'].startswith('COMPLETE_'):
            out['stages'].append(row)
            continue
        summary_path = service / 'benchmark/summary.json'
        groups = json.loads(summary_path.read_text()) if summary_path.exists() else []
        row['groups'] = len(groups)
        if groups:
            row['last'] = {key: round(value, 4) if isinstance(value, float) else value
                           for key, value in groups[-1].items()
                           if key in ['concurrency', 'repeat', 'throughput_frames_s', 'failed']}
        quality_path = directory / 'quality.json'
        if quality_path.exists():
            quality = json.loads(quality_path.read_text())['groups']
            row['wer_groups_complete'] = sum(g.get('wer', {}).get('evaluated') == 1088 for g in quality.values())
            row['quality_records'] = sum(len(g.get('records', [])) for g in quality.values())
            row['sim_records'] = sum(r.get('speaker_similarity') is not None
                                     for g in quality.values() for r in g.get('records', []))
        elif result['status'] == 'RUNNING':
            log = service / 'server.log'
            if stage not in {'vllm-quality', 'sglang-saved-quality'}:
                row['HTTP_200_headers'] = (log.read_text(errors='replace').count('POST /v1/audio/speech HTTP/1.1" 200')
                                           if log.exists() else 0)
        if result.get('error'):
            row['error'] = result['error']
        out['stages'].append(row)
    except json.JSONDecodeError:
        out['stages'].append({'stage': stage, 'status': 'result file being written'})
out['stages'] = [row for row in out['stages'] if row['status'] in {'RUNNING', 'result file being written'} or (row['status'] == 'STOPPED_FAILURE' and row['stage'] not in {'vllm-service', 'sglang'})]
print(json.dumps(out, ensure_ascii=False))
