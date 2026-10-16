"""Bounded process ownership and private-MPS cleanup; no GPU work at import."""
import csv
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from comparison_common import TOKEN_KEY, require, save


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'ppid':int(stat[1]),'starttime':int(stat[19]),'uid':Path(f'/proc/{pid}').stat().st_uid,'state':stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_environment(pid):
    try:
        entries=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        fields={k.decode(errors='replace'):v.decode(errors='replace') for part in entries if b'=' in part for k,v in [part.split(b'=',1)]}
        return {k:fields.get(k) for k in (TOKEN_KEY,'CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY')}
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return {}


def token_owned(record, process_env, token, earliest_starttime):
    return bool(record and record['uid']==os.getuid() and record['starttime']>=earliest_starttime and process_env.get(TOKEN_KEY)==token)


def discover_owned(token, earliest_starttime, known):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid=int(entry.name)
        record=identity(pid)
        if not record or record['uid']!=os.getuid():
            continue
        env=process_environment(pid)
        if token_owned(record,env,token,earliest_starttime):
            try:
                command=Path(f'/proc/{pid}/comm').read_text().strip()
            except FileNotFoundError:
                continue
            known[pid]={**record,'environment':env,'comm':command}
    return known


def gpu_processes(protocol):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    result=[]
    for row in csv.reader(text.splitlines()):
        if not row or row[0].strip()!=protocol['gpu_uuid']:
            continue
        memory=row[3].strip()
        result.append({'uuid':row[0].strip(),'pid':int(row[1]),'name':row[2].strip(),'memory_mib':int(memory) if memory.isdigit() else None})
    return result


def same_process(pid, record):
    current=identity(pid)
    return bool(current and current['starttime']==record['starttime'] and current['uid']==record['uid'] and current['state']!='Z')


def private_mps_pipes(known, token, gpu_uuid):
    result={}
    for record in known.values():
        env=record['environment']
        value=env.get('CUDA_MPS_PIPE_DIRECTORY')
        if not value or env.get(TOKEN_KEY)!=token or 'mps' not in record['comm']:
            continue
        path=Path(value)
        require(path.name=='pipe' and path.parent.name.startswith('vllm-omni-mps-'), 'refuse non-private MPS directory')
        require(env.get('CUDA_VISIBLE_DEVICES')==gpu_uuid,'MPS physical GPU differs')
        result[str(path)]={'CUDA_MPS_PIPE_DIRECTORY':str(path),'CUDA_MPS_LOG_DIRECTORY':env.get('CUDA_MPS_LOG_DIRECTORY') or str(path.parent/'log'),'CUDA_VISIBLE_DEVICES':gpu_uuid}
    return result


def run_monitored(command, env, directory, protocol, timeout_seconds):
    directory.mkdir(exist_ok=False)
    token=uuid.uuid4().hex
    env=dict(env, **{TOKEN_KEY:token})
    known={}
    process=None
    process_record=None
    started=time.monotonic()
    earliest=identity(os.getpid())['starttime']
    report={'status':'RUNNING','command':command,'environment':{k:v for k,v in env.items() if k.startswith(('DRAMA3_','CUDA_MPS_')) or k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_MODULES_CACHE','PYTHONDONTWRITEBYTECODE')},'token':token,'sampling_interval_seconds':1,'limitation':'One-second polling cannot exclude shorter interference or observe true instantaneous memory peaks. Only the selected physical UUID is checked; other GPU jobs are untouched.','cleanup':[]}
    save(directory/'command.json',command)
    save(directory/'result.json',report)
    error=None
    try:
        require(not gpu_processes(protocol),'selected GPU occupied before launch')
        actual_uuid=subprocess.check_output(['nvidia-smi','-i',str(protocol['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
        require(actual_uuid==protocol['gpu_uuid'],'GPU index/UUID mismatch')
        with (directory/'stdout.log').open('x') as log, (directory/'process-audit.jsonl').open('x') as audit:
            process=subprocess.Popen(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process_record=identity(process.pid)
            require(process_record is not None,'runner disappeared at launch')
            earliest=process_record['starttime']
            known[process.pid]={**process_record,'environment':{k:env.get(k) for k in (TOKEN_KEY,'CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY')},'comm':'owned runner'}
            report['runner']=process_record
            while True:
                discover_owned(token,earliest,known)
                rows=gpu_processes(protocol)
                audit.write(json.dumps({'elapsed_seconds':time.monotonic()-started,'gpu_processes':rows,'known_owned_pids':sorted(known)})+'\n')
                audit.flush()
                for row in rows:
                    pid=row['pid'];current=identity(pid)
                    if pid not in known:
                        discover_owned(token,earliest,known)
                    require(pid in known and (current is None or current['starttime']==known[pid]['starttime']), f'foreign or unattributed selected-GPU process {pid}')
                if process.poll() is not None:
                    require(process.returncode==0,f'command exited {process.returncode}')
                    break
                require(time.monotonic()-started<timeout_seconds,'watchdog expired')
                time.sleep(1)
    except BaseException as exc:  # noqa: BLE001 - record interruption before owned cleanup
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if process and process.poll() is None and process_record and same_process(process.pid,process_record):
            os.killpg(process.pid,signal.SIGTERM)
            report['cleanup'].append({'action':'SIGTERM own runner process group','pid':process.pid})
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                if same_process(process.pid,process_record):
                    os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=10)
        discover_owned(token,earliest,known)
        try:
            pipes=private_mps_pipes(known,token,protocol['gpu_uuid'])
            for pipe,mps_env in pipes.items():
                # Native close normally removes its private root. Never issue
                # quit without a token-owned MPS process and explicit socket.
                owners=[pid for pid,record in known.items() if record['environment'].get('CUDA_MPS_PIPE_DIRECTORY')==pipe and 'mps' in record['comm'] and same_process(pid,record)]
                if not owners or not Path(pipe).is_dir():
                    continue
                result=subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',env=env|mps_env,text=True,capture_output=True,timeout=15,check=False)
                report['cleanup'].append({'action':'quit private MPS','pipe':pipe,'owners':owners,'exit_code':result.returncode,'stderr':result.stderr})
                if result.returncode:
                    error=(error+'; ' if error else '')+'private MPS quit failed'
        except Exception as exc:  # noqa: BLE001 - cleanup failure must not skip remaining cleanup
            error=(error+'; ' if error else '')+f'MPS cleanup error: {exc}'
        # Daemons may have reparented. The inherited random token plus uid and
        # starttime proves ownership independently of their current parent.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            alive=[pid for pid,record in known.items() if same_process(pid,record)]
            for pid in alive:
                try:
                    os.kill(pid,sig)
                    report['cleanup'].append({'action':sig.name,'pid':pid,'starttime':known[pid]['starttime']})
                except ProcessLookupError:
                    pass
            deadline=time.monotonic()+5
            while time.monotonic()<deadline and any(same_process(pid,known[pid]) for pid in alive):
                time.sleep(.2)
        residual=[pid for pid,record in known.items() if same_process(pid,record)]
        if residual:
            error=(error+'; ' if error else '')+f'owned processes remained: {residual}'
        try:
            post=gpu_processes(protocol)
            if post:
                error=(error+'; ' if error else '')+'selected GPU not empty after cleanup'
        except Exception as exc:  # noqa: BLE001 - cleanup failure must not skip remaining cleanup
            post=None;error=(error+'; ' if error else '')+f'post GPU audit failed: {exc}'
        report.update(status='FAIL' if error else 'PASS',error=error,elapsed_seconds=time.monotonic()-started,exit_code=process.returncode if process else None,owned_processes=known,post_gpu=post)
        save(directory/'result.json',report)
    require(error is None,error)
    return report
