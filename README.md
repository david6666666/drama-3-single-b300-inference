# Drama 3 on one B300

Built with Fish Audio.

A specialized, single-process streaming inference runtime for
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3), plus reproducible
B300 comparisons with vLLM-Omni and SGLang-Omni.

**Development status:** baseline measurement and candidate validation are in
progress. The target is at least **2× the better upstream throughput at each
concurrency**. This is an acceptance target, not an achieved performance claim.

The implementation uses a Python admission loop, persistent request slots,
CUDA Graph decode and direct GPU operators. The engine does not instantiate
vLLM, SGLang or an Omni pipeline. Fish Speech supplies the checkpoint loader,
prompt format and codec; FlashInfer and low-level SGLang/FLA modules supply
operators. Rust is deferred until profiling demonstrates a control-loop cost
that warrants another language.

## Scope

- One B300, native FP8 expert weights, BF16 model activations and FP32 GDN state.
- Streaming mono PCM at 44.1 kHz; at most one inline voice reference.
- Temperature 1, top-p 0.9, top-k 30, up to 500 acoustic frames.
- Continuous batching at capacities 1–64; one model and one codec per process.
- No training, multi-GPU execution or new quantization scheme.

The benchmark contract and provenance are in [docs/draft.md](docs/draft.md).
Exact source revisions, model/dataset revisions and dependency snapshots are
under [evidence/20261014-b300](evidence/20261014-b300).

## Benchmark tools

Install the lightweight client and CPU tests:

```bash
uv venv --python 3.12 /path/to/benchmark-venv
uv pip install --python /path/to/benchmark-venv/bin/python -e '.[dev]'
/path/to/benchmark-venv/bin/pytest -q
```

Prepare the pinned Seed-TTS English subset with `scripts/prepare_corpus.py`.
Use `scripts/run_baseline.py --help` to launch a baseline from an explicitly
supplied source checkout and environment. The launcher records the source SHA,
command and readiness time, runs the client, and shuts down only its own process
group. Benchmark requests use the same waveform and transcript for both APIs.

```bash
drama3-bench --url http://127.0.0.1:18080 --backend sglang \
  --corpus /path/to/requests.jsonl --output /path/to/new-results \
  --concurrency 1 2 4 8 16 32 64 --requests 192 --repeats 3 \
  --warm-references --save-audio
```

Throughput counts returned audio samples / 2048 per measured batch second.
TTFA starts at dispatch and ends at the first PCM bytes. Mean RTF divides each
request's dispatch-to-last-audio time by its output audio duration. Failed
requests invalidate a run; they are preserved in raw timing records.

## Candidate runtime

The development environment uses the pinned SGLang CUDA dependencies recorded
in the evidence directory, plus the pinned Fish Speech checkout installed with
`--no-deps` to retain the tested Torch/CUDA stack. The model implementation is
still undergoing GPU validation; a successful import is not a readiness claim.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m drama3.server \
  --checkpoint /path/to/pinned-drama-3-snapshot \
  --fish-source /path/to/pinned-fish-speech --slots 64 --port 18082
```

`/health` becomes ready only after checkpoint loading, CUDA Graph capture and
codec warmup. Use `--backend standalone` with the same benchmark command.

`scripts/validate_model.py` compares operators and layers with Fish Speech's
reference math. `scripts/evaluate_audio.py` scores saved audio with the upstream
Seed-TTS WER calculation, Whisper-large-v3 and WavLM speaker similarity. Missing
quality coverage cannot produce a passing result.

## References

- [OmniRSI PR #9](https://github.com/ThinkFlowLab/OmniRSI/pull/9): Strata,
  vLLM-Omni and SGLang-Omni serving knowledge packs.
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and
  [SGLang-Omni](https://github.com/sgl-project/sglang-omni): main-branch baselines.
- [Fish Speech](https://github.com/fishaudio/fish-speech): reference architecture.
- [NVlabs/KDA](https://github.com/NVlabs/kda): numerical validation and kernel
  profiling workflow. Kernels here are not claimed to be KDA-generated.

See [NOTICE](NOTICE) and [Fish Audio's license](third_party_licenses/Fish-Audio-Community-License.txt)
for the Fish Audio materials and dependency attribution.
