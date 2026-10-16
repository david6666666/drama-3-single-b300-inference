"""Synthetic CPU coverage for final two-incomplete-row report and dry-run archive."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summary = load('summarize_matrix')
archive = load('archive_comparison')
checks = []


def check(name, okay):
    if not okay:
        raise AssertionError(name)
    checks.append({'name': name, 'passed': True})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bad(name, call):
    try:
        call()
    except (RuntimeError, FileNotFoundError, KeyError):
        check(name, True)
    else:
        raise AssertionError('accepted: ' + name)


with tempfile.TemporaryDirectory(prefix='drama3-final-partial-cpu-') as tmp:
    art = Path(tmp) / 'campaign'
    out, recovery, interrupted = art / summary.REPORT_ID, art / summary.RECOVERY, art / summary.INTERRUPTED
    out.mkdir(parents=True);recovery.mkdir();interrupted.mkdir()
    summary.OUT, summary.ART = out, art
    source = art / 'source-standalone/src/drama3/benchmark.py'
    source.parent.mkdir(parents=True);source.write_text('# synthetic client\n')
    record = {'bytes': source.stat().st_size, 'sha256': summary.digest(source)}
    protocol = {'concurrency': list(summary.CONCURRENCY), 'repeats': 3, 'requests': 1088,
                'frameworks': list(summary.BACKENDS), 'source_shas': {b: b + '-sha' for b in summary.BACKENDS}, 'corpus_sha256': 'synthetic'}
    write(art/'protocol.json',protocol)
    write(art/'frozen-inputs.json',{'files':{str(source):record}})
    write(art/'preparation.json',{'imports':{b:{'packages':{'torch':'synthetic'}} for b in summary.BACKENDS}})
    for root, plan, manifest in [(recovery,'matrix-protocol.json','matrix-frozen-inputs.json'),(interrupted,'protocol.json','frozen-inputs.json'),(out,'protocol.json','frozen-inputs.json')]:
        write(root/plan,{'attempt_id':root.name});write(root/manifest,{str(source):record})
    bindings = [('protocol_sha256',art/'protocol.json'),('manifest_sha256',art/'frozen-inputs.json'),
                ('matrix_protocol_sha256',recovery/'matrix-protocol.json'),('matrix_manifest_sha256',recovery/'matrix-frozen-inputs.json'),
                ('continuation_protocol_sha256',interrupted/'protocol.json'),('continuation_manifest_sha256',interrupted/'frozen-inputs.json'),
                ('remaining_protocol_sha256',out/'protocol.json'),('remaining_manifest_sha256',out/'frozen-inputs.json')]
    hashes = {key:summary.digest(path) for key,path in bindings}
    labels = [f'c{c}-r{r}' for c in summary.CONCURRENCY for r in range(3)]
    segments = [('standalone','standalone',recovery/'matrix/standalone',recovery/'matrix/standalone',labels),
                ('vllm','vllm',recovery/'matrix/vllm',interrupted/'vllm-quality',labels[:-1]),
                ('sglang_saved','sglang',interrupted/'sglang',out/'existing-quality',labels[:5]),
                ('sglang_remaining','sglang',out/'remaining',out/'remaining',labels[6:])]
    for segment,backend,service,result_root,wanted in segments:
        write(service/'service/launch.json',{'source_sha':protocol['source_shas'][backend],'source_status':'','benchmark_client_sha256':summary.digest(source)})
        write(service/'service/benchmark/summary.json',{'synthetic':True})
        pg,qg,quality_records = [],[],{}
        for label in wanted:
            c,repeat = map(int,label[1:].split('-r'))
            partial = segment=='sglang_saved' and label=='c2-r1'
            raw=service/'service/benchmark'/(label+'.jsonl');raw.write_text('{"synthetic":true}\n')
            pg.append({'label':label,'requests':1088,'jsonl_sha256':summary.digest(raw),'valid':not partial,'diagnostic_only':partial,
                       **{k:100+c+repeat for k in summary.PERF}})
            qg.append({'label':label,'requests':1088,'wer_percent':20+repeat if backend=='standalone' else 1+repeat,'sim_mean':60+repeat,
                       'scored_requests':1087 if partial else 1088,'complete':not partial})
            records=[{'request_id':f'{label}-{i:05d}','is_success':not(partial and i==351),'speaker_similarity':None if partial and i==351 else 60.0} for i in range(1088)]
            quality_records[label]={'records':records,'complete':not partial}
        write(result_root/'quality.json',{'groups':quality_records})
        perf={'status':'PASS','groups_checked':len(wanted),'requests_checked':len(wanted)*1088,'groups':pg,
              'all_wav_pcm_hashes_verified':True,'corpus_sha256':'synthetic','summary_sha256':summary.digest(service/'service/benchmark/summary.json')}
        quality={'status':'PASS','groups_checked':len(wanted),'requests_checked':len(wanted)*1088,'groups':qg,
                 'unfiltered_wer':True,'all_sim_records_present':True,'quality_sha256':summary.digest(result_root/'quality.json')}
        if segment=='vllm':perf.update(status='PASS_COMPLETED_GROUPS_ONLY',planned_groups=21,interrupted_group='c64-r2',full_matrix_valid=False)
        if segment=='sglang_saved':
            for audit in (perf,quality):audit.update(status='VALIDATED_WITH_KNOWN_PARTIAL_FAILURE',full_matrix_valid=False,complete_groups=4,partial_groups=1,failed_records=1)
            del perf['all_wav_pcm_hashes_verified'], quality['all_sim_records_present']
            perf.update(successful_audio_checked=5439,all_successful_wav_pcm_hashes_verified=True)
            quality.update(scored_requests=5439,all_successful_sim_records_present=True,known_failure_sim=None)
        stopped=segment in ('vllm','sglang_saved')
        write(service/'service-runner/result.json',{'status':'FAIL' if stopped else 'PASS','post_gpu':[],'error':'preserved failure' if stopped else None})
        write(result_root/'quality-runner/result.json',{'status':'PASS','post_gpu':[]})
        result={'status':{'vllm':'COMPLETE_VALID_EXISTING_GROUPS_AND_QUALITY','sglang_saved':'COMPLETE_VALID_SAVED_AUDIO_WITH_PRESERVED_FAILURE'}.get(segment,'COMPLETE_VALID_SERVICE_AND_QUALITY'),
                'attempt_id':summary.RECOVERY if segment=='standalone' else summary.INTERRUPTED if segment=='vllm' else summary.REPORT_ID,
                **hashes,'service':perf,'quality':quality}
        if stopped:
            write(service/'result.json',{'status':'STOPPED_FAILURE','error':'synthetic failure'})
            result.update(failed_service_result_sha256=summary.digest(service/'result.json'),failed_runner_result_sha256=summary.digest(service/'service-runner/result.json'))
        write(result_root/'result.json',result)
    write(art/'matrix/standalone/result.json',{'status':'STOPPED_FAILURE','groups':7,'fps':999999})
    write(art/'feasibility/sglang/result.json',{'fps':888888})
    write(art/'cpu-initial-probe.json',{'initial':True})
    (interrupted/'sglang/service/server.log').write_text('preserved HTTP500 no audio\n')
    for relative in ('audio/test.wav','checkpoint/model.bin','cache/result.json','tensor.pt.gz'):
        path=art/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'excluded')
    aggregate=lambda:summary.aggregate(out)
    data=aggregate()
    partial_result=json.loads((out/'existing-quality/result.json').read_text())
    check('partial_fixture_actual_keys_without_full_coverage_aliases',
          'all_wav_pcm_hashes_verified' not in partial_result['service']
          and 'all_sim_records_present' not in partial_result['quality']
          and 'successful_audio_checked' not in partial_result['quality']
          and partial_result['quality']['scored_requests']==5439)
    check('60full65280_plus1087partial66367',data['performance_groups']==data['quality_groups']==60 and data['complete_group_requests']==65280 and data['partial_scored_requests']==1087 and data['total_scored_requests']==66367)
    check('19formal2incomplete4diagnostic',len(data['rows'])==19 and len(data['incomplete_rows'])==2 and len(data['diagnostic_groups'])==4)
    check('failed_rows_have_no_formal_median',not any((r['framework'],r['concurrency']) in [('vllm',64),('sglang',2)] for r in data['rows']) and all(r['metrics'] is None for r in data['incomplete_rows']))
    d=next(d for d in data['diagnostic_groups'] if d['framework']=='sglang' and d['label']=='c2-r1')
    check('partial_diagnostic_explicit1087success1failure',not d['valid_complete_group'] and d['attempted_requests']==1088 and d['successful_requests']==d['scored_requests']==1087 and d['failed_requests']==1)
    check('unattempted_sglang_c2r2_not_replaced',data['not_run_groups']==[{'framework':'sglang','label':'c2-r2','status':'NOT_RUN','reason':'No retry or completion of failed C2 row authorized'}])
    check('c64_quality_comparison_unavailable',all(not r['quality_vs_vllm']['available'] for r in data['rows'] if r['concurrency']==64))
    check('three_run_medians_only',data['rows'][0]['metrics']['throughput_frames_s']=={'median':102,'min':101,'max':103})
    check('bad_quality_reported_without_acceptance',not data['rows'][0]['quality_vs_vllm']['median_within_reporting_bounds'] and data['optimization_acceptance_evaluated'] is False)
    sg=[r for r in data['rows'] if r['framework']=='sglang']
    check('SG_two_sessions_row_provenance',sg[0]['source_segment']=='sglang_saved' and all(r['source_segment']=='sglang_remaining' for r in sg[1:]) and data['sglang_server_lifecycles']['count']==2)
    check('aggregation_does_not_write',not(out/'matrix-summary.json').exists())
    bad('old_report_directory_refused',lambda:summary.aggregate(interrupted))
    path=out/'existing-quality/result.json';original=json.loads(path.read_text())
    mutations={
      'quality_not_finished_refused':lambda x:x.update(status='RUNNING'),
      'wrong_eighth_binding_refused':lambda x:x.update(remaining_manifest_sha256='wrong'),
      'partial_service_not_fullPASS':lambda x:x['service'].update(status='PASS'),
      'partial_quality_not_fullPASS':lambda x:x['quality'].update(status='PASS'),
      'partial_failure_count_preserved':lambda x:x['quality'].update(failed_records=0),
      'partial_audio_count_preserved':lambda x:x['quality'].update(scored_requests=5440),
      'partial_service_audio_count_preserved':lambda x:x['service'].update(successful_audio_checked=5440),
      'partial_successful_pcm_audit_required':lambda x:x['service'].pop('all_successful_wav_pcm_hashes_verified'),
      'partial_successful_sim_audit_required':lambda x:x['quality'].pop('all_successful_sim_records_present'),
      'partial_failure_sim_must_remain_null':lambda x:x['quality'].update(known_failure_sim=60),
      'missing_complete_group_refused':lambda x:x['service']['groups'].pop(0),
      'duplicate_quality_label_refused':lambda x:x['quality']['groups'].__setitem__(1,x['quality']['groups'][0]),
      'invented_c2r2_refused':lambda x:x['service']['groups'].append(dict(x['service']['groups'][-1],label='c2-r2')),
      'nonfinite_partial_metrics_refused':lambda x:x['service']['groups'][-1].update(throughput_frames_s=float('nan')),
      'changed_preserved_failure_refused':lambda x:x.update(failed_service_result_sha256='wrong'),
    }
    for name,mutate in mutations.items():
        changed=copy.deepcopy(original);mutate(changed);write(path,changed);bad(name,aggregate);write(path,original)
    qp=out/'existing-quality/quality.json';qr=json.loads(qp.read_text());altered=copy.deepcopy(qr)
    altered['groups']['c2-r1']['records'][351].update(is_success=True,speaker_similarity=60)
    write(qp,altered);changed=copy.deepcopy(original);changed['quality']['quality_sha256']=summary.digest(qp);write(path,changed)
    bad('failed_output_cannot_receive_fabricated_score',aggregate)
    write(qp,qr);write(path,original)
    remaining=out/'remaining/result.json';r=json.loads(remaining.read_text());changed=copy.deepcopy(r);changed['service']['groups'].append(dict(changed['service']['groups'][0],label='c2-r2'))
    write(remaining,changed);bad('remaining_cannot_retry_C2',aggregate);write(remaining,r)
    write(out/'matrix-summary.json',aggregate())
    destination=Path(tmp)/'archive';selection=archive.plan(art,destination,out);files={r['source']:r for r in selection['files']}
    check('archive_plan_no_writes',not destination.exists())
    check('all_three_failures_and_pilots_preserved',all(p in files for p in ('matrix/standalone/result.json','recovery-attempt-01/matrix/vllm/result.json','interrupted-matrix-report-01/sglang/result.json','feasibility/sglang/result.json','cpu-initial-probe.json')))
    check('failed_group_raw_and_partial_quality_preserved','interrupted-matrix-report-01/sglang/service/benchmark/c2-r1.jsonl' in files and 'remaining-sglang-report-01/existing-quality/quality.json' in files)
    check('raw_quality_logs_gzip',all(r['encoding']=='gzip' for p,r in files.items() if p.endswith(('.jsonl','.log','quality.json'))))
    check('worktrees_weights_audio_caches_excluded',not any(p.startswith(('source-','audio/','checkpoint/','cache/')) or p.endswith('.pt.gz') for p in files))
    sp=out/'matrix-summary.json';stored=json.loads(sp.read_text());write(sp,dict(stored,complete_matrix=True))
    bad('archive_false_complete_matrix_refused',lambda:archive.plan(art,destination,out));write(sp,stored)
    destination.mkdir();bad('archive_overwrite_refused',lambda:archive.plan(art,destination,out))
    check('archive_not_executed',list(destination.iterdir())==[])

report={'status':'CPU_SYNTHETIC_REMAINING_REPORT_CHECK_PASS','scope':'Synthetic fixtures only; no actual GPU performance inferred. archive() never called; no GPU/repository writes.',
        'checks':checks,'sources':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('summarize_matrix.py','archive_comparison.py','check_helpers.py')}}
(HERE/'helper-cpu-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks)}))
