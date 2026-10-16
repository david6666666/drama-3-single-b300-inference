"""Separately approved continuation of the frozen descriptive comparison."""
import argparse
import json
import sys
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parent
sys.path.insert(0, str(ATTEMPT.parent))

from comparison_common import (
    OUT,
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['standalone','vllm','sglang'],required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text())
    plan=json.loads((ATTEMPT/'matrix-protocol.json').read_text())
    root=ATTEMPT/'matrix'/args.backend
    service=root/'service';quality=root/'quality.json'
    labels=[f'c{c}-r{r}' for c in protocol['concurrency'] for r in range(3)]
    commands={'service':service_command(protocol,args.backend,'matrix',service),'quality':quality_command(protocol,service,labels,quality)}
    if not args.execute:
        print(json.dumps({'status':'DRY_RUN_NO_GPU','commands':commands},indent=2));return
    verify_execution_authorization(protocol)
    require(plan['status']=='FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL','matrix plan is not frozen')
    require(plan['base_protocol_sha256']==sha(OUT/'protocol.json') and plan['base_manifest_sha256']==sha(OUT/'frozen-inputs.json'),'base protocol changed')
    manifest=json.loads((ATTEMPT/'matrix-frozen-inputs.json').read_text())
    for name,value in manifest.items():
        require(Path(name).stat().st_size==value['bytes'] and sha(name)==value['sha256'],'matrix source changed: '+name)
    approval=json.loads((ATTEMPT/'execution-approval.json').read_text())
    require(approval.get('approved_by')=='root' and approval.get('gpu_execution') is True and approval.get('protocol_sha256')==sha(ATTEMPT/'matrix-protocol.json') and approval.get('manifest_sha256')==sha(ATTEMPT/'matrix-frozen-inputs.json'),'matrix approval mismatch')
    require(plan['attempt_id']==ATTEMPT.name,'attempt identity mismatch')
    for backend in protocol['order']:
        require(sha(OUT/'feasibility'/backend/'result.json')==plan['feasibility_result_sha256'][backend],'referenced feasibility changed')
        feasibility=json.loads((OUT/'feasibility'/backend/'result.json').read_text())
        require(feasibility['status']=='COMPLETE_VALID_SERVICE_AND_QUALITY','all frameworks must complete feasibility first')
        require(feasibility['protocol_sha256']==sha(OUT/'protocol.json') and feasibility['manifest_sha256']==sha(OUT/'frozen-inputs.json'),'feasibility source identity differs')
    for prior in protocol['order'][:protocol['order'].index(args.backend)]:
        require(json.loads((ATTEMPT/'matrix'/prior/'result.json').read_text())['status']=='COMPLETE_VALID_SERVICE_AND_QUALITY','preceding framework matrix incomplete')
    root.parent.mkdir(exist_ok=True);root.mkdir(exist_ok=False)
    report={'status':'RUNNING','attempt_id':ATTEMPT.name,'backend':args.backend,'protocol_sha256':sha(OUT/'protocol.json'),'manifest_sha256':sha(OUT/'frozen-inputs.json'),'matrix_protocol_sha256':sha(ATTEMPT/'matrix-protocol.json'),'matrix_manifest_sha256':sha(ATTEMPT/'matrix-frozen-inputs.json'),'commands':commands}
    save(root/'result.json',report)
    try:
        run_monitored(commands['service'],environment(protocol,args.backend,gpu=True),root/'service-runner',protocol,plan['watchdog_seconds']['service'])
        report['service']=validate_service(service,protocol['corpus'],protocol['concurrency'],3)
        save(root/'result.json',report)
        run_monitored(commands['quality'],environment(protocol,'standalone',gpu=True,quality=True),root/'quality-runner',protocol,plan['watchdog_seconds']['quality'])
        report['quality']=validate_quality(quality,service,protocol['corpus'],labels)
        report['status']='COMPLETE_VALID_SERVICE_AND_QUALITY'
    except BaseException as exc:
        report.update(status='STOPPED_FAILURE',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root/'result.json',report)
    print(json.dumps({'status':report['status'],'backend':args.backend}))


if __name__=='__main__':
    main()
