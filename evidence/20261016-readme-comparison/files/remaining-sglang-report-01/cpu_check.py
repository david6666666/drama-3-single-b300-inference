"""CPU-only CLI checks; no old frozen file is changed."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert HERE.name == 'remaining-sglang-report-01'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checks = []
for name in ['continuation.py', 'prepare_protocol.py', 'cpu_check.py']:
    ast.parse((HERE / name).read_text())
    checks.append('AST:' + name)
plan = json.loads((HERE / 'protocol.json').read_text())
assert plan['status'] == 'DRAFT_CPU_ONLY'
checks.append('CPU-only draft')
assert plan['saved_labels'] == ['c1-r0', 'c1-r1', 'c1-r2', 'c2-r0', 'c2-r1']
assert plan['remaining_labels'] == [f'c{c}-r{r}' for c in [4, 8, 16, 32, 64] for r in range(3)]
checks.append('5 saved labels and15 never-attempted groups')
for stage in ['existing-quality', 'remaining']:
    command = [sys.executable, str(HERE / 'continuation.py'), '--stage', stage]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['status'] == 'DRY_RUN_NO_GPU'
    q = data['commands']['quality']; labels = q[q.index('--labels')+1:q.index('--output')]
    assert labels == plan['saved_labels' if stage == 'existing-quality' else 'remaining_labels']
    assert ('service' in data['commands']) == (stage == 'remaining')
    if stage == 'remaining':
        cmd = data['commands']['service']
        assert cmd[cmd.index('--concurrency')+1:cmd.index('--skip-reference-warmup')] == ['4', '8', '16', '32', '64']
        assert cmd[cmd.index('--repeats')+1] == '3' and cmd[cmd.index('--requests')+1] == '1088'
        checks.append('only unattempted C4..64 x3 with1088 requests')
    checks.append('default dry-run:' + stage)
    denied = subprocess.run(command + ['--execute'], capture_output=True, text=True)
    assert denied.returncode != 0 and 'remaining continuation is CPU-only draft' in denied.stderr
    assert not (HERE / stage).exists()
    checks.append('draft execution refused before output/GPU:' + stage)
report = {'status': 'CPU_REMAINING_PASS', 'checks': len(checks), 'details': checks,
          'gpu_workloads_launched': False, 'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ['continuation.py', 'prepare_protocol.py', 'cpu_check.py']}}
(HERE / 'cpu-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
