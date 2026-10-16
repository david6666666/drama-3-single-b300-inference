"""Fixed workload and source controls for the latest-source comparison."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
VENVS = Path('/home/zjy/code/hsliu/venvs')
OLD_PROTOCOL = Path('/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/native-row-reuse-service/protocol.json')
SOURCE_SHAS = {
    'standalone': '33f6d7237196520be9f457d86df161264cbb087e',
    'vllm': 'aa9c9a2d2ce3ff8640f1b4ab8937a6194b0ead4d',
    'sglang': '2c23b68f4f763c14cfb845eadb4883a5e791b701',
}
TOKEN_KEY = 'OMNI_COMPARISON_RUN_TOKEN'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git_identity(path):
    return {
        'sha': subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git','-C',str(path),'status','--porcelain'], text=True),
        'branch': subprocess.check_output(['git','-C',str(path),'branch','--show-current'], text=True).strip(),
    }


def environment(protocol, backend, *, gpu=False, run_token=None, quality=False):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DRAMA3_','VLLM_','SGLANG_','CUDA_MPS_','NCCL_')) and k not in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS','CUBLAS_WORKSPACE_CONFIG',TOKEN_KEY)}
    source = Path(protocol['frameworks'][backend]['source'])
    paths = [str(source/'src') if backend=='standalone' else str(source), str(OUT/'source-standalone/src'), protocol['fish_source']]
    env.update(PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(protocol['gpu']) if gpu else '', OMP_NUM_THREADS='8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_MODULES_CACHE=protocol['quality_hf_modules'], PYTHONUNBUFFERED='1')
    if backend == 'standalone' and not quality:
        env.update(protocol['standalone_flags'])
    if run_token:
        env[TOKEN_KEY] = run_token
    return env


def service_command(protocol, backend, phase, directory):
    config = protocol['frameworks'][backend]
    concurrency = [64] if phase=='feasibility' else protocol['concurrency']
    repeats = 1 if phase=='feasibility' else 3
    command = [str(VENVS/'codex-drama3-standalone/bin/python'), str(OUT/'source-standalone/scripts/run_baseline.py'), '--backend', backend, '--source', config['source'], '--venv', config['venv'], '--bench-python', str(VENVS/'codex-drama3-standalone/bin/python'), '--checkpoint', protocol['checkpoint'], '--corpus', protocol['corpus'], '--output', str(directory), '--gpu', str(protocol['gpu']), '--port', str(config['port']), '--capacity', str(config['capacity']), '--requests','1088','--repeats',str(repeats),'--concurrency',*map(str,concurrency),'--skip-reference-warmup']
    if backend=='standalone':
        command += ['--fish-source', protocol['fish_source']]
    return command


def quality_command(protocol, service_dir, labels, output):
    return [str(VENVS/'codex-drama3-standalone/bin/python'),str(OUT/'source-standalone/scripts/evaluate_audio.py'),'--upstream',protocol['quality_upstream'],'--assets',protocol['quality_assets'],'--corpus',protocol['corpus'],'--benchmark',str(service_dir/'benchmark'),'--labels',*labels,'--output',str(output),'--batch-size','8']


def verify_plan(protocol):
    require(protocol['source_shas']==SOURCE_SHAS, 'source SHA contract changed')
    for backend, expected in SOURCE_SHAS.items():
        identity = git_identity(OUT/f'source-{backend}')
        require(identity['sha']==expected and not identity['status'], f'{backend} source changed or dirty')
        require(identity['branch'].startswith('codex-'), f'{backend} source branch is not isolated')
    old = json.loads(OLD_PROTOCOL.read_text())
    flags = dict(old['common_flags'], DRAMA3_MIXED_ROW_REUSE='0')
    require(protocol['standalone_flags']==flags, 'standalone profile changed')
    require(protocol['cpu_affinity']==old['cpu_affinity']==list(range(240)), 'CPU affinity contract changed')
    require(sorted(os.sched_getaffinity(0))==protocol['cpu_affinity'], 'actual CPU affinity mismatch')
    require(protocol['gpu']==6 and protocol['gpu_uuid']=='GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf', 'GPU contract changed')
    require(protocol['requests']==1088 and protocol['concurrency']==[1,2,4,8,16,32,64] and protocol['repeats']==3, 'workload shape changed')
    require(sha(protocol['corpus'])==protocol['corpus_sha256']=='08612ee05ab0dd8a762109978b5f230e5b0311a1dfde87f03a4572434dff2dac','corpus changed')
    for key in ('fish_source','quality_upstream'):
        identity=git_identity(protocol[key])
        require(identity['sha']==protocol[key+'_sha'] and not identity['status'], key+' changed')
    require(sha(protocol['quality_assets'])==protocol['quality_assets_sha256'],'quality assets manifest changed')
    require({k:v['capacity'] for k,v in protocol['frameworks'].items()}=={'standalone':64,'vllm':128,'sglang':64},'capacity changed')
    for backend in SOURCE_SHAS:
        probe=json.loads((OUT/f'cpu-probes/{backend}-imports.json').read_text())
        require(probe['status']=='CPU_IMPORT_PASS' and not probe['cuda_initialized'], backend+' CPU import failed')


def verify_execution_authorization(protocol):
    verify_plan(protocol)
    require(protocol['status']=='FROZEN_AWAITING_EXPLICIT_EXECUTION_APPROVAL', 'protocol not frozen')
    manifest=json.loads((OUT/'frozen-inputs.json').read_text())
    for path, record in manifest['files'].items():
        require(Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'frozen input changed: '+path)
    preparation=json.loads((OUT/'preparation.json').read_text())
    for name, expected in preparation['checkpoint_files'].items():
        path=Path(name);stat=path.stat()
        require({'resolved':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}==expected, 'checkpoint file identity changed: '+name)
    approval_path=OUT/'execution-approval.json'
    require(approval_path.is_file(), 'CPU-only preparation: root execution approval is absent')
    approval=json.loads(approval_path.read_text())
    require(approval.get('protocol_sha256')==sha(OUT/'protocol.json') and approval.get('manifest_sha256')==sha(OUT/'frozen-inputs.json') and approval.get('approved_by')=='root' and approval.get('gpu_execution') is True,'approval is not bound to this exact protocol/manifest')
    return approval
