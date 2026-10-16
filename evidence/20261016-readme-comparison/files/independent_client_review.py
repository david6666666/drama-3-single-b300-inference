"""CPU-only public payload audit; never starts a serving engine or CUDA."""
import hashlib
import inspect
import json
import os
import re
import subprocess
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
ART = Path(__file__).resolve().parent
ROOT = Path('/home/zjy/code/hsliu/drama-3-single-b300-inference')
CORPUS = Path('/home/zjy/code/hsliu/artifacts/drama3-b300-20261014/seedtts-en1088-codes/requests.jsonl')
RAW = CORPUS.parent.parent / 'seedtts-en1088/requests.jsonl'
OUT = ART / 'independent-client-review.json'
result = {'status': 'RUNNING', 'scope': 'CPU payload/schema/normalization only; no HTTP server, model execution, GPU, or new performance data', 'checks': [], 'sources': {}}

def check(name, condition, **details):
    result['checks'].append({'name': name, 'passed': bool(condition), **details})
    if not condition:
        raise AssertionError(name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

try:
    import torch
    import drama3.benchmark as bench
    import sglang_omni.serve.protocol as protocol
    import sglang_omni.serve.speech_service as service
    import sglang_omni.models.fishaudio_drama3.config as config
    import sglang_omni.models.fishaudio_drama3.speech_request as speech

    for name, module, expected in [('benchmark', bench, ART/'source-standalone/src'), ('protocol', protocol, ART/'source-sglang'), ('service', service, ART/'source-sglang'), ('config', config, ART/'source-sglang'), ('speech_request', speech, ART/'source-sglang')]:
        path = Path(inspect.getfile(module)).resolve()
        check('bound_source/'+name, path.is_relative_to(expected))
        result['sources'][name] = {'path':str(path), 'sha256':sha(path)}
    for name in ['source-standalone','source-sglang','source-vllm']:
        result['sources'][name] = {'sha':subprocess.check_output(['git','-C',str(ART/name),'rev-parse','HEAD'],text=True).strip()}
    check('cuda_not_initialized_before', not torch.cuda.is_initialized())
    samples = [json.loads(s) for s in CORPUS.read_text().splitlines()]
    raw = [json.loads(s) for s in RAW.read_text().splitlines()]
    check('corpus_1088', len(samples)==len(raw)==1088)
    check('raw_coded_identity', all(all(a[k]==b[k] for k in ['id','text','reference_text','reference_audio','reference_sha256']) for a,b in zip(samples,raw,strict=True)))
    validator = service.SpeechRequestValidator(default_model='fishaudio/drama-3', supported_speech_fields=config.Drama3PipelineConfig.supported_speech_fields)
    preprocessor = object.__new__(speech.Drama3SpeechPreprocessor)
    integer_count=0
    all_generation={'model':'fishaudio/drama-3','response_format':'pcm','stream':True,'stream_format':'audio','max_new_tokens':500,'temperature':1.0,'top_p':0.9,'top_k':30}
    body_hashes={k:hashlib.sha256() for k in ['standalone','vllm','sglang']}
    for sample in samples:
        sid=sample['id']; codes=sample['reference_codes']
        bodies={k:bench.request_body(sample,k,500) for k in body_hashes}
        for key,body in bodies.items():
            body_hashes[key].update(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        check(sid+'/all_sampling_and_text', all(all(b[k]==v for k,v in all_generation.items()) and b['input']==sample['text'] for b in bodies.values()))
        check(sid+'/standalone_vllm_equal', bodies['standalone']==bodies['vllm'])
        check(sid+'/code_layout_range', len(codes)==10 and 1<=len(codes[0])<=1292 and all(len(row)==len(codes[0]) and all(type(v) is int and 0<=v<(4096 if i==0 else 1024) for v in row) for i,row in enumerate(codes)))
        integer_count+=sum(map(len,codes))
        check(sid+'/codes_transcript_exact', bodies['vllm']['ref_audio_codes']==codes and bodies['vllm']['ref_text']==sample['reference_text'] and bodies['sglang']['references']==[{'vq_codes':codes,'text':sample['reference_text']}])
        check(sid+'/encoded_only', 'ref_audio' not in bodies['vllm'] and 'references' not in bodies['vllm'] and set(bodies['sglang']['references'][0])=={'vq_codes','text'} and 'ref_audio_codes' not in bodies['sglang'] and 'ref_audio' not in bodies['sglang'])
        prepared=validator.parse_generation_request(bodies['sglang'])
        expected=[{'vq_codes':codes,'text':sample['reference_text']}]
        check(sid+'/public_schema_normalize_exact', prepared.reference_descriptors==expected and [r.model_dump(exclude_none=True) for r in prepared.request.references]==expected)
        lowered=service.build_speech_prompt(prepared.request,prepared.reference_descriptors)
        check(sid+'/lowered_prompt_exact', lowered=={'text':sample['text'],'references':expected})
        sampled=service.build_sampling_params(prepared.request)
        check(sid+'/lowered_sampling_exact', all(getattr(sampled,k)==v for k,v in {'max_new_tokens':500,'temperature':1.0,'top_p':0.9,'top_k':30}.items()))
        tensor=preprocessor.reference_audio_codes(prepared.reference_descriptors[0],0)
        check(sid+'/preprocessor_cpu_int64_exact', tensor.device.type=='cpu' and tensor.dtype==torch.int64 and tensor.tolist()==codes)
    result['corpus']={'path':str(CORPUS),'sha256':sha(CORPUS),'raw_path':str(RAW),'raw_sha256':sha(RAW),'requests':len(samples),'integers_checked':integer_count,'body_stream_sha256':{k:v.hexdigest() for k,v in body_hashes.items()}}
    readme=(ROOT/'README.md').read_text()
    blocks=re.findall(r'```bash\n(.*?)```',readme,re.S)
    for i,block in enumerate(blocks):
        parsed=subprocess.run(['bash','-n'],input=block,text=True,capture_output=True)
        check('readme/bash_syntax/'+str(i), parsed.returncode==0, stderr=parsed.stderr)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',readme):
        if '://' not in target and not target.startswith('#'):
            links.append(target);check('readme/local_link/'+target,(ROOT/target.split('#')[0]).exists())
    result['readme']={'path':str(ROOT/'README.md'),'sha256':sha(ROOT/'README.md'),'bash_blocks':len(blocks),'local_links':links,'review':'No blocking quickstart or architecture defect found. Installer requires uv and creates Python3.12 only for absent venv; model download separate; example uses that venv executable. Curl body matches SpeechRequest defaults. Cards mark mixed/pipeline/async as opt-in. Values remain placeholders until measured.'}
    check('cuda_not_initialized_after', not torch.cuda.is_initialized())
    result['status']='CPU_PUBLIC_ADAPTER_PASS'
except Exception as error:
    import traceback
    result['status']='CPU_PUBLIC_ADAPTER_FAIL'
    result['error']=f'{type(error).__name__}: {error}'
    result['traceback']=traceback.format_exc()
finally:
    result['driver_sha256']=sha(Path(__file__))
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','error'] if k in result}|{'checks':len(result['checks']),'output':str(OUT)},ensure_ascii=False))
raise SystemExit(result['status']!='CPU_PUBLIC_ADAPTER_PASS')
