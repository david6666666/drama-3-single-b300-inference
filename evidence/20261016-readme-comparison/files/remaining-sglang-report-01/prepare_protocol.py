"""CPU-only freeze of a continuation that never retries an attempted service group."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RECOVERY = BASE / 'recovery-attempt-01'
INTERRUPTED = BASE / 'interrupted-matrix-report-01'
sys.path.insert(0, str(BASE))
from comparison_common import require, save, sha, verify_execution_authorization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires empty CUDA_VISIBLE_DEVICES')
    base = json.loads((BASE / 'protocol.json').read_text())
    verify_execution_authorization(base)
    failed = INTERRUPTED / 'sglang'
    require(json.loads((failed / 'result.json').read_text())['status'] == 'STOPPED_FAILURE', 'original SGLang failure missing')
    require(json.loads((failed / 'service-runner/result.json').read_text())['post_gpu'] == [], 'original cleanup incomplete')
    summary = json.loads((failed / 'service/benchmark/summary.json').read_text())
    saved = ['c1-r0', 'c1-r1', 'c1-r2', 'c2-r0', 'c2-r1']
    require([f"c{x['concurrency']}-r{x['repeat']}" for x in summary] == saved, 'saved group identity differs')
    require(all(x['valid'] for x in summary[:4]) and not summary[4]['valid'] and summary[4]['failed'] == 1, 'saved failure changed')
    plan = {
        'status': 'FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL' if args.freeze else 'DRAFT_CPU_ONLY',
        'attempt_id': HERE.name,
        'purpose': 'Preserve both upstream service failures, score every available SGLang output, and measure only concurrency groups never attempted. No optimization or retry of failed/existing groups.',
        'protocol_sha256': sha(BASE / 'protocol.json'), 'manifest_sha256': sha(BASE / 'frozen-inputs.json'),
        'matrix_protocol_sha256': sha(RECOVERY / 'matrix-protocol.json'), 'matrix_manifest_sha256': sha(RECOVERY / 'matrix-frozen-inputs.json'),
        'continuation_protocol_sha256': sha(INTERRUPTED / 'protocol.json'), 'continuation_manifest_sha256': sha(INTERRUPTED / 'frozen-inputs.json'),
        'order': ['existing-quality', 'remaining'], 'saved_labels': saved,
        'remaining_concurrency': [4, 8, 16, 32, 64], 'remaining_labels': [f'c{c}-r{r}' for c in [4, 8, 16, 32, 64] for r in range(3)],
        'never_retry': 'Do not rerun C1 or any C2 request/group. C2-r1 remains invalid; C2-r2 remains NOT_RUN. vLLM c64-r2 remains incomplete.',
        'lifecycle_boundary': 'The unattempted C4..C64 groups use a fresh SGLang server, original capture/startup, and original max(8,C) warmups. RNG and prior service history differ from the interrupted lifecycle. This is an explicitly segmented campaign, not one uninterrupted server for all21 groups. Source, native kernels, model, sampling, capacity and all request controls remain unchanged.',
        'inherited_controls': 'Exact base source/venv/corpus/checkpoint/Fish/quality assets, physical GPU6 UUID, affinity0..239, OMP8, SGLang capacity64, original defaults and FP8/top8, maxframes500/temp1/topp.9/topk30, freshTCP/no retries/full tail and all audio are unchanged. Reuse original run_baseline.py and evaluate_audio.py; no package/source edits. UUID-scoped30s monitoring and token/starttime cleanup unchanged.',
        'partial_quality_policy': 'Pass all1088 source records of c2-r1 to the original scorer. Preserve exact failed HTTP500 record with is_success=false/error and SIM=null; score1087 existing audio outputs. Record skipped1/complete=false. Conditional1087 WER/SIM are diagnostic only, never complete1088 quality or formal performance. Validate all4352 outputs from4 full groups normally. No request filtering or replacement.',
        'formal_reporting': 'If remaining15 groups finish, 19 formal three-repeat rows are available. SGLang C2 and vLLM C64 stay failed/incomplete. There are60 complete valid groups(65280 outputs), plus1087 valid audio outputs from one invalid group, total66367 scored audio outputs. SG C2-r2 NOT_RUN. Three complete groups outside formal rows are diagnostic-only. Never claim all63 groups passed.',
        'preserved_failure': {'sglang_audit': str(INTERRUPTED / 'sglang-stop-audit.json'), 'sglang_audit_sha256': sha(INTERRUPTED / 'sglang-stop-audit.json'), 'vllm_audit_sha256': sha(RECOVERY / 'vllm-stop-audit.json')},
        'watchdog_seconds': {'service': 14400, 'quality': 14400},
        'failure_policy': 'Any new request/model/monitor/validation failure stops. No automatic retry, tuning, outlier removal or threshold change.',
        'requires': 'New root execution-approval.json bound to this exact protocol.json and frozen-inputs.json. Prior approvals are insufficient.',
    }
    save(HERE / 'protocol.json', plan)
    if not args.freeze:
        print(json.dumps({'status': plan['status'], 'protocol': str(HERE / 'protocol.json')}))
        return
    cpu = json.loads((HERE / 'cpu-check.json').read_text())
    require(cpu['status'] == 'CPU_REMAINING_PASS', 'controller CPU checks missing')
    for name, expected in cpu['source_sha256'].items():
        require(sha(HERE / name) == expected, 'CPU-checked source changed: ' + name)
    audit = json.loads((HERE / 'saved-service-audit.json').read_text())
    require(audit['status'] == 'VALIDATED_WITH_KNOWN_PARTIAL_FAILURE' and audit['groups_checked'] == 5 and audit['successful_audio_checked'] == 5439 and audit['failed_records'] == 1 and audit['full_matrix_valid'] is False, 'saved-service validator audit differs')
    require(json.loads((HERE / 'validator-cpu-check.json').read_text())['status'] == 'CPU_SAVED_PARTIAL_VALIDATORS_PASS', 'validator CPU checks missing')
    paths = list(HERE.glob('*.py')) + [HERE / 'protocol.json', HERE / 'cpu-check.json', HERE / 'saved-service-audit.json', HERE / 'validator-cpu-check.json']
    paths += list(HERE.glob('*review.json')) + list(HERE.glob('*cpu-check.json'))
    paths += [RECOVERY / 'process_control_recovery.py', RECOVERY / 'vllm-stop-audit.json', RECOVERY / 'matrix/standalone/result.json', INTERRUPTED / 'vllm-quality/result.json', INTERRUPTED / 'sglang-stop-audit.json', INTERRUPTED / 'sglang-continuation-cpu-assessment.json']
    paths += [RECOVERY / 'matrix/vllm/result.json', RECOVERY / 'matrix/vllm/service-runner/result.json']
    paths += [failed / x for x in ['result.json', 'service-runner/result.json', 'service/benchmark/manifest.json', 'service/benchmark/summary.json']]
    paths += [failed / 'service/benchmark' / f'{label}.jsonl' for label in saved]
    paths += [failed / 'service/benchmark' / f'c{c}-warmup.json' for c in [1, 2]]
    manifest = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(set(paths))}
    save(HERE / 'frozen-inputs.json', manifest)
    print(json.dumps({'status': plan['status'], 'files': len(manifest), 'protocol_sha256': sha(HERE / 'protocol.json'), 'manifest_sha256': sha(HERE / 'frozen-inputs.json')}))


if __name__ == '__main__':
    main()
