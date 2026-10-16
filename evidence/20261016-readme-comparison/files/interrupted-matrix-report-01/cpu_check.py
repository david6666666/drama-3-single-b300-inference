"""Small CPU-only checks of the continuation's CLI and unchanged workload."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checks = []
for name in ['continuation.py', 'prepare_protocol.py', 'cpu_check.py']:
    ast.parse((HERE / name).read_text())
    checks.append('AST:' + name)
plan = json.loads((HERE / 'protocol.json').read_text())
assert plan['status'] == 'DRAFT_CPU_ONLY'
checks.append('CPU-only draft')
assert len(plan['vllm_completed_labels']) == 20 and plan['vllm_completed_labels'][-1] == 'c64-r1'
assert len(plan['sglang_labels']) == 21 and plan['sglang_labels'][-1] == 'c64-r2'
checks.append('20 existing vLLM and 21 planned SGLang labels')
for stage in ['vllm-quality', 'sglang']:
    command = [sys.executable, str(HERE / 'continuation.py'), '--stage', stage]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['status'] == 'DRY_RUN_NO_GPU'
    q = data['commands']['quality']; labels = q[q.index('--labels')+1:q.index('--output')]
    assert labels == plan['vllm_completed_labels' if stage == 'vllm-quality' else 'sglang_labels']
    assert ('service' in data['commands']) == (stage == 'sglang')
    checks.append('default dry-run:' + stage)
    denied = subprocess.run(command + ['--execute'], capture_output=True, text=True)
    assert denied.returncode != 0 and 'continuation is CPU-only draft' in denied.stderr
    assert not (HERE / stage).exists()
    checks.append('draft execution refused before output/GPU:' + stage)
report = {'status': 'CPU_CONTINUATION_PASS', 'checks': len(checks), 'details': checks,
          'gpu_workloads_launched': False, 'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ['continuation.py', 'prepare_protocol.py', 'cpu_check.py']}}
(HERE / 'cpu-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
