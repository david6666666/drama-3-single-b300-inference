"""No CUDA or nvidia-smi execution: recovery command/ownership negative checks."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ATTEMPT=Path(__file__).resolve().parent
BASE=ATTEMPT.parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(ATTEMPT))
import process_control_recovery as pc
from comparison_common import save, service_command, sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    protocol=json.loads((BASE/'protocol.json').read_text())
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    trees={key:ast.parse(path.read_text()) for key,path in [('old',BASE/'process_control.py'),('new',ATTEMPT/'process_control_recovery.py')]}
    functions={key:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)} for key,tree in trees.items()}
    for name in ('identity','process_environment','token_owned','discover_owned','same_process','private_mps_pipes'):
        check(name+' unchanged',functions['old'][name]==functions['new'][name])
    with patch.object(pc.subprocess,'check_output',return_value=protocol['gpu_uuid']+', 123, python, 400\nGPU-other, 456, foreign, 200\n') as query:
        result=pc.gpu_processes(protocol)
        check('only selected rows',len(result)==1 and result[0]['pid']==123)
        check('UUID command',query.call_args.args[0][1:3]==['-i',protocol['gpu_uuid']])
        check('bounded30s',query.call_args.kwargs['timeout']==30)
    with patch.object(pc.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('mock-query',30)) as query:
        try:pc.gpu_processes(protocol)
        except subprocess.TimeoutExpired:check('query timeout propagates once',query.call_count==1)
        else:raise AssertionError('timeout suppressed')
    for wrong in [f'5, {protocol["gpu_uuid"]}', '6, GPU-wrong']:
        with tempfile.TemporaryDirectory() as temporary, patch.object(pc,'gpu_processes',return_value=[]), patch.object(pc,'discover_owned',side_effect=lambda token,earliest,known:known), patch.object(pc.subprocess,'check_output',return_value=wrong) as query, patch.object(pc.subprocess,'Popen') as launch:
            target=Path(temporary)/'attempt'
            try:pc.run_monitored(['never-launch'],{},target,protocol,30)
            except RuntimeError as error:check('wrong device rejected '+wrong,'GPU index/UUID mismatch' in str(error))
            else:raise AssertionError('bad device accepted')
            check('bad device no launch '+wrong,launch.call_count==0)
            check('preflight UUID+index',query.call_args.args[0][1:4]==['-i',protocol['gpu_uuid'],'--query-gpu=index,uuid'] and query.call_args.kwargs['timeout']==30)
            check('failure persisted '+wrong,json.loads((target/'result.json').read_text())['status']=='FAIL')
    spec=importlib.util.spec_from_file_location('recovery_runner_cpu_check',ATTEMPT/'run_matrix.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with patch.object(module,'verify_execution_authorization',return_value={}),patch.object(module,'run_monitored') as launch,patch.object(sys,'argv',['run_matrix.py','--backend','standalone','--execute']):
        try:module.main()
        except RuntimeError as error:check('draft refuses execution','not frozen' in str(error))
        else:raise AssertionError('draft allowed')
        check('draft never launches',launch.call_count==0)
    for backend in protocol['order']:
        dry=subprocess.run([sys.executable,str(ATTEMPT/'run_matrix.py'),'--backend',backend],check=True,text=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        value=json.loads(dry.stdout)
        check(backend+' default no GPU',value['status']=='DRY_RUN_NO_GPU')
        expected=service_command(protocol,backend,'matrix',ATTEMPT/'matrix'/backend/'service')
        check(backend+' unchanged controls',value['commands']['service']==expected)
        check(backend+' complete21 scorelabels',len(value['commands']['quality'][value['commands']['quality'].index('--labels')+1:value['commands']['quality'].index('--output')])==21)
    check('no real matrix output created',not (ATTEMPT/'matrix').exists())
    files=['process_control_recovery.py','run_matrix.py','prepare_recovery.py','cpu_check.py']
    for name in files:ast.parse((ATTEMPT/name).read_text())
    save(ATTEMPT/'cpu-check.json',{'status':'CPU_RECOVERY_PASS','checks':len(checks),'items':checks,'source_sha256':{name:sha(ATTEMPT/name) for name in files},'gpu_queries_or_launches':0,'old_frozen_inputs_untouched':True})
    print(json.dumps({'status':'CPU_RECOVERY_PASS','checks':len(checks)}))


if __name__=='__main__':
    main()
