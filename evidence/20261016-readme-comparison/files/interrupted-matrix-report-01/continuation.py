"""Explicitly preserve the failed matrix row; score saved groups and measure SGLang."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RECOVERY = BASE / 'recovery-attempt-01'
sys.path[:0] = [str(HERE), str(RECOVERY), str(BASE)]

from comparison_common import (
    environment,
    quality_command,
    require,
    save,
    service_command,
    sha,
    verify_execution_authorization,
)
from process_control_recovery import run_monitored
from validate_results import validate_quality, validate_service


def bindings():
    return {
        'protocol_sha256': sha(BASE / 'protocol.json'),
        'manifest_sha256': sha(BASE / 'frozen-inputs.json'),
        'matrix_protocol_sha256': sha(RECOVERY / 'matrix-protocol.json'),
        'matrix_manifest_sha256': sha(RECOVERY / 'matrix-frozen-inputs.json'),
        'continuation_protocol_sha256': sha(HERE / 'protocol.json'),
        'continuation_manifest_sha256': sha(HERE / 'frozen-inputs.json'),
    }


def authorize(base, plan):
    verify_execution_authorization(base)
    require(plan['status'] == 'FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL', 'continuation is CPU-only draft')
    require(plan['attempt_id'] == HERE.name, 'continuation identity changed')
    for path, expected in json.loads((HERE / 'frozen-inputs.json').read_text()).items():
        require(Path(path).stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'continuation input changed: ' + path)
    for key, actual in bindings().items():
        if key.startswith('continuation_'):
            continue
        require(plan[key] == actual, 'inherited protocol/manifest changed: ' + key)
    approval = json.loads((HERE / 'execution-approval.json').read_text())
    require(approval.get('approved_by') == 'root' and approval.get('gpu_execution') is True
            and approval.get('protocol_sha256') == sha(HERE / 'protocol.json')
            and approval.get('manifest_sha256') == sha(HERE / 'frozen-inputs.json'), 'separate continuation approval mismatch')
    failed = RECOVERY / 'matrix/vllm'
    require(json.loads((failed / 'result.json').read_text())['status'] == 'STOPPED_FAILURE', 'vLLM failure must remain preserved')
    require(json.loads((failed / 'service-runner/result.json').read_text())['post_gpu'] == [], 'failed service cleanup incomplete')
    require(json.loads((RECOVERY / 'matrix/standalone/result.json').read_text())['status'] == 'COMPLETE_VALID_SERVICE_AND_QUALITY', 'standalone matrix incomplete')
    for backend, expected in plan['feasibility_result_sha256'].items():
        path = BASE / 'feasibility' / backend / 'result.json'
        require(sha(path) == expected and json.loads(path.read_text())['status'] == 'COMPLETE_VALID_SERVICE_AND_QUALITY', 'feasibility changed: ' + backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['vllm-quality', 'sglang'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = json.loads((BASE / 'protocol.json').read_text())
    plan = json.loads((HERE / 'protocol.json').read_text())
    root = HERE / args.stage
    service = RECOVERY / 'matrix/vllm/service' if args.stage == 'vllm-quality' else root / 'service'
    quality = root / 'quality.json'
    labels = plan['vllm_completed_labels'] if args.stage == 'vllm-quality' else plan['sglang_labels']
    commands = {'quality': quality_command(base, service, labels, quality)}
    if args.stage == 'sglang':
        commands['service'] = service_command(base, 'sglang', 'matrix', service)
    if not args.execute:
        print(json.dumps({'status': 'DRY_RUN_NO_GPU', 'stage': args.stage, 'commands': commands}, indent=2))
        return
    authorize(base, plan)
    if args.stage == 'sglang':
        prior = json.loads((HERE / 'vllm-quality/result.json').read_text())
        require(prior['status'] == 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY', 'score all existing vLLM groups first')
        require(all(prior[k] == v for k, v in bindings().items()), 'prior continuation identity differs')
    root.mkdir(exist_ok=False)
    report = {'status': 'RUNNING', 'attempt_id': HERE.name, 'stage': args.stage,
              'backend': 'vllm' if args.stage == 'vllm-quality' else 'sglang', **bindings(), 'commands': commands}
    if args.stage == 'vllm-quality':
        report.update(failed_service_result_sha256=sha(RECOVERY / 'matrix/vllm/result.json'),
                      failed_runner_result_sha256=sha(RECOVERY / 'matrix/vllm/service-runner/result.json'))
    save(root / 'result.json', report)
    try:
        if args.stage == 'vllm-quality':
            from validate_existing_groups import validate_existing_groups
            report['service'] = validate_existing_groups(service, base['corpus'])
        else:
            run_monitored(commands['service'], environment(base, 'sglang', gpu=True), root / 'service-runner', base, plan['watchdog_seconds']['service'])
            report['service'] = validate_service(service, base['corpus'], base['concurrency'], 3)
        save(root / 'result.json', report)
        run_monitored(commands['quality'], environment(base, 'standalone', gpu=True, quality=True), root / 'quality-runner', base, plan['watchdog_seconds']['quality'])
        report['quality'] = validate_quality(quality, service, base['corpus'], labels)
        report['status'] = 'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY' if args.stage == 'vllm-quality' else 'COMPLETE_VALID_SERVICE_AND_QUALITY'
    except BaseException as exc:
        report.update(status='STOPPED_FAILURE', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root / 'result.json', report)
    print(json.dumps({'status': report['status'], 'stage': args.stage}))


if __name__ == '__main__':
    main()
