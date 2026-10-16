"""GPU-free provenance and compatibility check in an existing final-path venv."""
import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['standalone', 'vllm', 'sglang'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--fish', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU probe requires CUDA_VISIBLE_DEVICES empty')
    modules = {
        'standalone': ['drama3', 'drama3.benchmark', 'drama3.server', 'drama3.model', 'fish_speech', 'flashinfer', 'sglang'],
        'vllm': ['drama3.benchmark', 'vllm', 'vllm.entrypoints.cli.main', 'vllm_omni', 'vllm_omni.entrypoints.cli.main', 'vllm_omni.model_executor.models.fish_omni.fish_omni_model'],
        'sglang': ['drama3.benchmark', 'sglang_omni', 'sglang_omni.cli', 'sglang_omni.models.fishaudio_drama3.config', 'sglang_omni.models.fishaudio_drama3.speech_request', 'sglang_omni.models.fishaudio_drama3.sglang_model', 'sglang'],
    }[args.backend]
    result = {'status': 'RUNNING', 'backend': args.backend, 'executable': sys.executable, 'prefix': sys.prefix, 'python': sys.version, 'sys_path': sys.path, 'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','HF_MODULES_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE']}, 'modules': {}, 'packages': {}, 'errors': []}
    for name in ['torch','vllm','vllm-omni','sglang','sglang-omni','flashinfer-python','triton','transformers','numpy','aiohttp','fastapi','uvicorn','nvidia-cutlass-dsl']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    for name in modules:
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve() if getattr(module, '__file__', None) else None
            expected = args.client / 'src' if name.startswith('drama3') else args.source if name.startswith(('vllm_omni','sglang_omni')) else args.fish if name.startswith('fish_speech') else None
            okay = expected is None or (path is not None and path.is_relative_to(expected.resolve()))
            result['modules'][name] = {'file': str(path) if path else None, 'path': list(getattr(module, '__path__', [])), 'expected_root': str(expected) if expected else None, 'expected_root_matches': okay}
            if not okay:
                result['errors'].append(f'{name}: imported outside bound source')
        except Exception as error:
            result['modules'][name] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            result['errors'].append(f'{name}: {type(error).__name__}: {error}')
    torch = importlib.import_module('torch')
    result['cuda_initialized'] = torch.cuda.is_initialized()
    if result['cuda_initialized']:
        result['errors'].append('CUDA unexpectedly initialized')
    result['status'] = 'CPU_IMPORT_PASS' if not result['errors'] else 'CPU_IMPORT_FAIL'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status':result['status'],'errors':result['errors']},indent=2))
    raise SystemExit(bool(result['errors']))


if __name__ == '__main__':
    main()
