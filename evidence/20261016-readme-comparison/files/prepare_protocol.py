"""CPU-only fixed comparison plan; freezing never authorizes a GPU launch."""
import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from comparison_common import (
    OLD_PROTOCOL,
    OUT,
    SOURCE_SHAS,
    VENVS,
    git_identity,
    require,
    save,
    sha,
    verify_plan,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    old=json.loads(OLD_PROTOCOL.read_text())
    keys=['gpu','gpu_uuid','checkpoint','corpus','corpus_sha256','fish_source','fish_source_sha','quality_upstream','quality_upstream_sha','quality_assets','quality_assets_sha256','quality_hf_modules','cpu_affinity']
    p={key:old[key] for key in keys}
    p.update(status='FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL' if args.freeze else 'DRAFT_CPU_ONLY',created_utc=datetime.now(UTC).isoformat(),source_shas=SOURCE_SHAS,requests=1088,concurrency=[1,2,4,8,16,32,64],repeats=3,omp_threads=8,standalone_flags=dict(old['common_flags'],DRAMA3_MIXED_ROW_REUSE='0'),order=['standalone','vllm','sglang'])
    p['frameworks']={backend:{'source':str(OUT/f'source-{backend}'),'venv':str(VENVS/f'codex-drama3-{backend}'),'capacity':128 if backend=='vllm' else 64,'port':18146+i} for i,backend in enumerate(p['order'])}
    p['phases']={'feasibility':'For each framework serially: one C64 x 1088 service run, preserve all WAVs, then complete WER/SIM. All three must have valid complete outputs and complete scoring before matrix.','matrix':'Subsequent separately approved continuation: C1/2/4/8/16/32/64 x 3 repetitions, one server per framework configuration for all 21 groups, all WAVs and full quality. This initial runner executes feasibility only.'}
    p['objective']='Descriptive latest-source comparison; framework implementation/configuration is the comparison variable. Success means complete valid measurements and full scores, not an optimization improvement.'
    p['continuation']='No throughput or quality-score threshold filters comparison. Slower performance or worse WER/SIM is reported. Request failure, missing/corrupt audio, incomplete scoring, source drift or observed foreign selected-GPU process stops; no automatic config change/retry/optimization.'
    p['quality_reporting']='Use fixed historical scorer/assets on every unfiltered request. Report WER and SIM versus same-concurrency vLLM; previously accepted +1 percentage point WER and -2 SIM are reporting bounds, not continuation thresholds.'
    p['benchmark']={'max_frames':500,'temperature':1.0,'top_p':0.9,'top_k':30,'warmup':'max(8, concurrency) before each concurrency; omitted from measured groups','connection':'fresh TCP per request, no automatic retries','reference_codes':'same 1088 preencoded references; no extra reference warmup','audio':'complete mono PCM16 at 44100 Hz; one frame = 2048 samples','tail':'All requests/full drain included. No trimming, length matching, output filtering or EOS speculation.'}
    p['metrics']={'throughput_frames_s':'sum(audio_samples / 2048) / full group makespan','ttfa_p50_p95':'linear-interpolated percentiles of dispatch-to-first received audio chunk','rtf_mean':'arithmetic mean of (last_audio_time - dispatch_time) / audio_duration'}
    p['comparison_scope']='Latest frozen source on existing final-path environments, not latest rebuilt dependencies. Same workload/client/GPU/CPU affinity; native framework capacity, scheduler and codec differences remain explicit. No equal-output-length or identical RNG schedule claim.'
    p['framework_differences']={'standalone':'Historical 28-flag control profile plus row reuse disabled, native FP8 top8, GDN BF16; not bare-command FP32 state default. Codec BF16 and private async capture/thread.','vllm':'Latest fish_omni.yaml, max_num_seqs 128 in both stages, private MPS, chunked prefill/admission settings preserved, stateful FP16 codec. Existing distribution metadata version may name old editable revision; actual imported package/model/CLI paths point to the pinned new source. CPU import success does not prove GPU runtime compatibility.','sglang':'Latest Drama3 config with max_running_requests 64; existing SGLang/FlashInfer dependencies, native FP16 window codec, defaults otherwise unchanged.'}
    p['ownership']='Only GPU6 UUID is monitored. Empty preflight; random inherited token + UID/PID/starttime identifies own descendants/private MPS daemons. Poll once/sec, stop on observed foreign/query error. Only own identities are signalled; private MPS quit requires explicit observed token-owned pipe/UUID. Never global MPS quit or signals to other GPUs.'
    p['measurement_limits']='One-second process polling cannot exclude shorter interference. Generation sampling is not reset between groups; numerical quality and output lengths are measured, not assumed. Client does not expose native EOS reason. Preparation and readiness are separate from benchmark makespan.'
    p['watchdog_seconds']={'service_feasibility':3600,'quality_feasibility':3600}
    verify_plan(p)
    client=json.loads((OUT/'independent-client-review.json').read_text())
    require(client['status']=='CPU_PUBLIC_ADAPTER_PASS','client contract failed')
    paths=[Path(v['venv']) for v in p['frameworks'].values()]+[OUT,Path(p['checkpoint']),Path(p['quality_hf_modules'])]
    mounts={str(path):subprocess.check_output(['findmnt','-T',str(path),'-J'],text=True) for path in paths}
    probes={b:json.loads((OUT/f'cpu-probes/{b}-imports.json').read_text()) for b in p['order']}
    # Snapshot revision and exact local inode/size/mtime identities. No redundant
    # multi-gigabyte checkpoint rehash is claimed; checkpoint content was unchanged.
    checkpoint={str(path):{'resolved':str(path.resolve()),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'inode':path.stat().st_ino} for path in sorted(Path(p['checkpoint']).rglob('*')) if path.is_file()}
    save(OUT/'preparation.json',{'status':'CPU_PREPARATION_PASS','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'source_identities':{b:git_identity(v['source']) for b,v in p['frameworks'].items()},'imports':probes,'mounts':mounts,'disk_free_bytes':shutil.disk_usage(OUT).free,'checkpoint_files':checkpoint,'checkpoint_identity_scope':'Pinned HF revision and local resolved file/stat identities; no new full-weight SHA pass.','client_review_sha256':sha(OUT/'independent-client-review.json')})
    save(OUT/'protocol.json',p)
    if args.freeze:
        files=list(OUT.glob('*.py'))+[OUT/'protocol.json',OUT/'preparation.json',OUT/'independent-client-review.json',OUT/'validator-existing-records-check.json',OUT/'cpu-controller-check.json',OLD_PROTOCOL,Path(p['corpus']),Path(p['quality_assets'])]+list((OUT/'cpu-probes').glob('*-imports.json'))
        for backend in p['order']:
            root=Path(p['frameworks'][backend]['source'])
            names=subprocess.check_output(['git','-C',str(root),'ls-files','-z'],text=True).split('\0')
            files += [root/name for name in names if name and Path(name).suffix in ('.py','.yaml','.yml','.toml','.json') and (root/name).is_file()]
        manifest={'files':{str(path):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(set(files))}}
        save(OUT/'frozen-inputs.json',manifest)
        print(json.dumps({'status':p['status'],'protocol_sha256':sha(OUT/'protocol.json'),'manifest_sha256':sha(OUT/'frozen-inputs.json'),'files':len(manifest['files'])}))
    else:
        print(json.dumps({'status':p['status'],'protocol':str(OUT/'protocol.json')}))


if __name__=='__main__':
    main()
