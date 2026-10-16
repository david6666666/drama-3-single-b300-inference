"""Freeze the minimal post-failure continuation; never launches a GPU workload."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RECOVERY = BASE / 'recovery-attempt-01'
sys.path.insert(0, str(BASE))
from comparison_common import require, save, sha, verify_execution_authorization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires empty CUDA_VISIBLE_DEVICES')
    base = json.loads((BASE / 'protocol.json').read_text())
    verify_execution_authorization(base)
    failed = RECOVERY / 'matrix/vllm'
    require(json.loads((failed / 'result.json').read_text())['status'] == 'STOPPED_FAILURE', 'original failure missing')
    require(json.loads((failed / 'service-runner/result.json').read_text())['post_gpu'] == [], 'original cleanup incomplete')
    labels = [f'c{c}-r{r}' for c in base['concurrency'] for r in range(3)]
    summary = json.loads((failed / 'service/benchmark/summary.json').read_text())
    require([f"c{x['concurrency']}-r{x['repeat']}" for x in summary] == labels[:-1], 'completed groups differ')
    plan = {
        'status': 'FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL' if args.freeze else 'DRAFT_CPU_ONLY',
        'attempt_id': HERE.name,
        'purpose': 'Finish a descriptive latest-source comparison after an observed vLLM service timeout; no optimization, service retry, or replacement of failed observations.',
        'protocol_sha256': sha(BASE / 'protocol.json'), 'manifest_sha256': sha(BASE / 'frozen-inputs.json'),
        'matrix_protocol_sha256': sha(RECOVERY / 'matrix-protocol.json'), 'matrix_manifest_sha256': sha(RECOVERY / 'matrix-frozen-inputs.json'),
        'inherited_controls': 'All model/source/venv/corpus/CPU/GPU6/profile/capacity/sampling/warmup/fresh-TCP/tail/quality controls are unchanged from the exact base and recovery protocols. No install or source edit. Reuse immutable inputs; no copy of 7313 source files.',
        'order': ['vllm-quality', 'sglang'], 'vllm_completed_labels': labels[:-1], 'sglang_labels': labels,
        'vllm_service': str(failed / 'service'), 'vllm_failed_group': 'c64-r2',
        'preserved_failure': {'audit': str(RECOVERY / 'vllm-stop-audit.json'), 'audit_sha256': sha(RECOVERY / 'vllm-stop-audit.json'), 'result_sha256': sha(failed / 'result.json'), 'runner_result_sha256': sha(failed / 'service-runner/result.json')},
        'feasibility_result_sha256': {b: sha(BASE / 'feasibility' / b / 'result.json') for b in base['order']},
        'watchdog_seconds': {'service': 14400, 'quality': 14400},
        'allowed_actions': 'Validate and score all 20 saved full vLLM groups (21760 requests), then run SGLang once as its original 21 groups and score all 22848 requests. Same UUID-scoped 30s monitoring and token/starttime/private-MPS cleanup. Any further failure stops, no automatic retry.',
        'reporting': 'The vLLM C64 row remains failed/incomplete because only two of three planned repeats completed. Never calculate its formal three-repeat median or quality/performance verdict; retain two completed C64 observations as diagnostic-only. Fully measured rows: standalone 7, vLLM 6, SGLang 7. Maximum complete groups are 62 plus one failed/incomplete group; never call all 63 PASS.',
        'success_condition': 'Complete the permitted saved-group quality and SGLang records with no hidden filtering, full audio/identity/quality verification, and explicit preservation of the incomplete vLLM row. Performance or quality score cannot trigger tuning or retry.',
        'requires': 'Separate root execution-approval.json bound to this new protocol.json and frozen-inputs.json. Existing approvals alone cannot execute this continuation.',
    }
    save(HERE / 'protocol.json', plan)
    if not args.freeze:
        print(json.dumps({'status': plan['status'], 'protocol': str(HERE / 'protocol.json')}))
        return
    require(json.loads((HERE / 'cpu-check.json').read_text())['status'] == 'CPU_CONTINUATION_PASS', 'controller CPU checks missing')
    cpu = json.loads((HERE / 'cpu-check.json').read_text())
    for name, expected in cpu['source_sha256'].items():
        require(sha(HERE / name) == expected, 'CPU-checked source changed: ' + name)
    validator = json.loads((HERE / 'vllm-completed-service-audit.json').read_text())
    require(validator['status'] == 'PASS_COMPLETED_GROUPS_ONLY' and validator['groups_checked'] == 20 and validator['full_matrix_valid'] is False, 'partial service audit missing')
    paths = list(HERE.glob('*.py')) + [HERE / 'protocol.json', HERE / 'cpu-check.json']
    paths += list(HERE.glob('*review.json')) + list(HERE.glob('*cpu-check.json')) + [HERE / 'vllm-completed-service-audit.json']
    paths += [RECOVERY / 'process_control_recovery.py', RECOVERY / 'vllm-stop-audit.json', RECOVERY / 'matrix/standalone/result.json']
    paths += [failed / x for x in ['result.json', 'service-runner/result.json', 'stop-observation.json', 'service/benchmark/manifest.json', 'service/benchmark/summary.json']]
    paths += [failed / 'service/benchmark' / f'{label}.jsonl' for label in labels[:-1]]
    paths += [failed / 'service/benchmark' / f'c{c}-warmup.json' for c in base['concurrency']]
    paths += [BASE / 'feasibility' / b / 'result.json' for b in base['order']]
    manifest = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(set(paths))}
    save(HERE / 'frozen-inputs.json', manifest)
    print(json.dumps({'status': plan['status'], 'files': len(manifest), 'protocol_sha256': sha(HERE / 'protocol.json'), 'manifest_sha256': sha(HERE / 'frozen-inputs.json')}))


if __name__ == '__main__':
    main()
