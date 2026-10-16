"""One fixed feasibility campaign. Default prints commands and does not query GPUs."""
import argparse
import json

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
from process_control import run_monitored
from validate_results import validate_quality, validate_service


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--backend',choices=['standalone','vllm','sglang'],required=True)
    args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text())
    backend=args.backend
    phase='feasibility'
    root=OUT/phase/backend
    service=root/'service'
    quality=root/'quality.json'
    commands={'service':service_command(protocol,backend,phase,service),'quality':quality_command(protocol,service,['c64-r0'],quality)}
    if not args.execute:
        print(json.dumps({'status':'DRY_RUN_NO_GPU','backend':backend,'commands':commands},indent=2));return
    verify_execution_authorization(protocol)
    for prior in protocol['order'][:protocol['order'].index(backend)]:
        previous=json.loads((OUT/phase/prior/'result.json').read_text())
        require(previous['status']=='COMPLETE_VALID_SERVICE_AND_QUALITY','preceding framework incomplete')
    root.parent.mkdir(exist_ok=True)
    root.mkdir(exist_ok=False)
    report={'status':'RUNNING','backend':backend,'protocol_sha256':sha(OUT/'protocol.json'),'manifest_sha256':sha(OUT/'frozen-inputs.json'),'commands':commands}
    save(root/'result.json',report)
    try:
        run_monitored(commands['service'],environment(protocol,backend,gpu=True),root/'service-runner',protocol,protocol['watchdog_seconds']['service_feasibility'])
        report['service']=validate_service(service,protocol['corpus'],[64],1)
        save(root/'result.json',report)
        # The server/private MPS must already be fully stopped by run_monitored.
        run_monitored(commands['quality'],environment(protocol,'standalone',gpu=True,quality=True),root/'quality-runner',protocol,protocol['watchdog_seconds']['quality_feasibility'])
        report['quality']=validate_quality(quality,service,protocol['corpus'],['c64-r0'])
        report['status']='COMPLETE_VALID_SERVICE_AND_QUALITY'
    except BaseException as exc:
        report.update(status='STOPPED_FAILURE',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root/'result.json',report)
    print(json.dumps({'status':report['status'],'backend':backend}))


if __name__=='__main__':
    main()
