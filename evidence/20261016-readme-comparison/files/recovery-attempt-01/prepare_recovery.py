"""CPU-only recovery freeze. No GPU query or workload can be launched here."""
import argparse
import json
import os
import sys
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parent
BASE = ATTEMPT.parent
sys.path.insert(0, str(BASE))
from comparison_common import require, save, sha, verify_execution_authorization


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation needs empty CUDA_VISIBLE_DEVICES')
    base=json.loads((BASE/'protocol.json').read_text())
    verify_execution_authorization(base)
    old=json.loads((BASE/'matrix-protocol.json').read_text())
    results={}
    for b in base['order']:
        path=BASE/'feasibility'/b/'result.json';x=json.loads(path.read_text())
        require(x['status']=='COMPLETE_VALID_SERVICE_AND_QUALITY','feasibility incomplete')
        results[b]=sha(path)
    stopped=json.loads((BASE/'matrix-stop-audit.json').read_text())
    require(stopped['status']=='MATRIX_STOPPED_MONITOR_QUERY_TIMEOUT' and stopped['post_gpu']==[],'old failure or cleanup changed')
    p=dict(old,status='FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL' if args.freeze else 'DRAFT_CPU_ONLY',attempt_id=ATTEMPT.name,feasibility_result_sha256=results)
    p.update(recovery_reason='Original attempt stopped after seven complete groups because a full-device nvidia-smi compute-app query exceeded 10 seconds. The cause of that delay is not established; it is not classified as a model failure or observed external GPU use.',preserved_failure={'path':str(BASE/'matrix-stop-audit.json'),'sha256':sha(BASE/'matrix-stop-audit.json'),'raw_runner_result_sha256':sha(BASE/'matrix/standalone/service-runner/result.json')},monitor_change={'selection':'Both nvidia-smi commands explicitly select frozen GPU6 UUID. Preflight also verifies the returned index is still 6.','query_timeout_seconds':30,'previous_query_timeout_seconds':10,'interval_seconds':1,'failure_policy':'Any timeout, query error, observed foreign process or unexpected index/UUID still aborts and cleans only owned identities. No query retry or interpolation. Actual successful-poll timestamps expose gaps.','scope':'Single selected-device query avoids requesting unrelated device process lists; it does not prove the original timeout cause or guarantee a timeout cannot recur.'},measurement_policy='Rerun standalone all 21 groups from the beginning, then vLLM then SGLang; fresh output paths. Do not splice any old completed or interrupted groups into recovery results. Old full C64 feasibility is reused by exact result hashes. Source, dependencies, benchmark parameters, request corpus, GPU6, affinity and 29 standalone flags remain the original frozen controls.',requires='Separate root approval in this attempt execution-approval.json bound to matrix-protocol and matrix-frozen-input hashes. Original approval alone cannot authorize recovery.')
    save(ATTEMPT/'matrix-protocol.json',p)
    if args.freeze:
        check=json.loads((ATTEMPT/'cpu-check.json').read_text());require(check['status']=='CPU_RECOVERY_PASS','CPU checks not passed')
        for name, expected in check['source_sha256'].items():
            require(sha(ATTEMPT/name)==expected,'CPU-checked source changed: '+name)
        paths=sorted(ATTEMPT.glob('*.py'))+[ATTEMPT/'matrix-protocol.json',ATTEMPT/'cpu-check.json',BASE/'matrix-stop-audit.json',BASE/'matrix/standalone/service-runner/result.json']+[BASE/'feasibility'/b/'result.json' for b in base['order']]
        paths += list(ATTEMPT.glob('*review.json'))+list(ATTEMPT.glob('*cpu-check.json'))
        manifest={str(path):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(set(paths))}
        save(ATTEMPT/'matrix-frozen-inputs.json',manifest)
        print(json.dumps({'status':p['status'],'files':len(manifest),'protocol_sha256':sha(ATTEMPT/'matrix-protocol.json'),'manifest_sha256':sha(ATTEMPT/'matrix-frozen-inputs.json')}))
    else:
        print(json.dumps({'status':p['status'],'protocol':str(ATTEMPT/'matrix-protocol.json')}))


if __name__=='__main__':
    main()
