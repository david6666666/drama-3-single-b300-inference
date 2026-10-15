# Reproduce the PR #8933 comparison

Use one B300 for each serving run and run the two servers sequentially on the
same device. The measured source revisions are vLLM-Omni
`d6ed4f4844fbc41573ffc03247e5b2ff0a38d322` and standalone `dcd20d3`.
The latter includes the prefill state-isolation fix. Later optional codec and
GDN experiments are separate from this comparison.

Use the pinned Fish Speech revision `541566932abbaebcb0a40e562a1f846738383f38`
and model snapshot `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`.
The standalone dependency installer is `scripts/install_runtime.py`;
`requirements-runtime.lock` records the tested environment. The upstream
environment is recorded in `evidence/20261014-b300/vllm-environment.txt`.

The examples below use the original machine's paths. Each output directory
must be new. Keep virtual environments, model weights and JIT caches on local
storage; verify with `findmnt -T`. Reuse caches and record preparation separately.

```bash
task_root=/home/zjy/code/hsliu
run_root="$task_root/artifacts/drama3-reproduction"
candidate_source="$task_root/drama3-sources/candidate-prefill192"
vllm_source="$task_root/drama3-sources/vllm-omni"
fish_source="$task_root/drama3-sources/fish-speech"
runtime_env="$task_root/venvs/codex-drama3-standalone"
vllm_env="$task_root/venvs/codex-drama3-vllm"
checkpoint=/home/zjy/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/e0a9218aa48888270f06ef1ed9ea67622c0e17d2

"$runtime_env/bin/python" scripts/prepare_corpus.py \
  --revision 81d1901582dee1293a537a6d945d084301712c41 \
  --count 1088 --output "$run_root/corpus"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
"$runtime_env/bin/python" scripts/encode_references.py \
  --checkpoint "$checkpoint" --fish-source "$fish_source" \
  --corpus "$run_root/corpus/requests.jsonl" --output "$run_root/codes"

"$runtime_env/bin/python" scripts/run_baseline.py \
  --backend vllm --source "$vllm_source" --venv "$vllm_env" \
  --bench-python "$runtime_env/bin/python" --checkpoint "$checkpoint" \
  --corpus "$run_root/codes/requests.jsonl" --output "$run_root/vllm" \
  --gpu 0 --port 18080 --capacity 128 --requests 1088 --repeats 3 \
  --concurrency 1 2 4 8 16 32 64 --skip-reference-warmup

DRAMA3_GDN_STATE_DTYPE=bfloat16 \
DRAMA3_PACKED_MOE_PROJECTION=1 \
DRAMA3_PREFILL_CONV=triton \
DRAMA3_PREFILL_MOE_TUNED=1 \
DRAMA3_ADMISSION_MIN_BATCH=6 \
DRAMA3_PREFILL_BUCKET_192=1 \
"$runtime_env/bin/python" scripts/run_baseline.py \
  --backend standalone --source "$candidate_source" --venv "$runtime_env" \
  --bench-python "$runtime_env/bin/python" --checkpoint "$checkpoint" \
  --fish-source "$fish_source" --corpus "$run_root/codes/requests.jsonl" \
  --output "$run_root/standalone" --gpu 0 --port 18082 --capacity 64 \
  --requests 1088 --repeats 3 --concurrency 1 2 4 8 16 32 64 \
  --skip-reference-warmup

"$runtime_env/bin/python" scripts/report_results.py \
  --vllm "$run_root/vllm/benchmark/summary.json" \
  --candidate "$run_root/standalone/benchmark/summary.json" \
  --output "$run_root/comparison.md"

"$runtime_env/bin/python" scripts/plot_results.py \
  --vllm "$run_root/vllm" --candidate "$run_root/standalone" \
  --output "$run_root/figures"
```

The candidate source path above must point to a clean checkout at `dcd20d3`;
it is not a directory installed by the dependency script. Verify both source
SHAs before starting. Do not inherit other `DRAMA3_*` experiment variables.
The launcher records the effective environment, source status, readiness,
client hash and complete command. Different upstream capacities are recorded
deliberately: vLLM uses its stronger measured configuration, while offered
concurrency is identical and never exceeds 64.

Use `scripts/evaluate_audio.py` with the pinned upstream Seed-TTS evaluators
and `scripts/report_quality.py` for WER/SIM. Required groups are the first
repetition at every concurrency plus the other two C64 repetitions. Keep all
1088 rows, including high-WER outliers. Quality scores and performance are
separate results; neither a kernel timing nor a partial run can pass the target.

The original corpus hash is
`08612ee05ab0dd8a762109978b5f230e5b0311a1dfde87f03a4572434dff2dac`.
The manifest includes absolute reference paths, so relocating the corpus changes
its byte hash. Both runs in a reproduction must use exactly the same generated
manifest; reference audio hashes, sample IDs and encoded integers remain the
portable workload identity.
