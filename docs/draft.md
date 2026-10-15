# Drama 3 on one B300

The target is a standalone streaming TTS runtime for `fishaudio/drama-3`.
The updated acceptance target is at least twice the measured vLLM-Omni
throughput after PR #8933 at each concurrency, subject to output quality. Until
the complete comparison exists, the target remains unverified.

## Baseline contract

- One NVIDIA B300, GPU 0, all inference and codec stages on that GPU.
- Freeze both upstream main revisions, checkpoint revision, dependency lists,
  driver, CPU affinity, dataset and sampling before measuring.
- Concurrency: 1, 2, 4, 8, 16, 32, 64. Closed-loop request replacement.
- Native FP8 routed experts and BF16 remaining model weights. The new upstream
  deployment uses BF16 recurrent state; the candidate currently retains FP32.
- Temperature 1.0, top-p 0.9, top-k 30, maximum 500 generated frames.
- Seed-TTS English voice cloning. PR #8933 adds pre-encoded VQ references;
  the new comparison sends both servers identical reference codes and transcript.
  Codes are prepared once with the pinned Fish codec in FP32. The old inline
  waveform tests are retained separately; inline vLLM requests encode references
  per request, so warming them does not create a production pre-encoded baseline.
- Report native default configuration first. Separately record any tuned
  baseline; compare the candidate against the stronger valid configuration.
- Throughput is total returned acoustic frames divided by the measured batch
  makespan, excluding warmup. Do not count EOS or network chunks as frames.
  With this codec, 2048 waveform samples correspond to one acoustic frame;
  report fractional frame equivalents if the endpoint trims waveform samples.
- TTFA is request dispatch to receipt of the first nonempty playable PCM bytes.
- RTF is request dispatch to last audio bytes divided by output audio duration.
- Record failures, empty output, truncation, sample rate, audio hashes and
  per-request timings. No silent filtering of failed requests.
- Separate environment preparation, startup, capture, warmup and measurement.
- One feasibility request before load; three repetitions for comparison claims.

## Implementation directions

1. Profile the stronger baseline to split GPU compute, launch, scheduling,
   transport and codec cost. No language-based speedup assumptions.
2. Consolidate the single-GPU model and codec execution into one specialized
   runtime, using existing validated kernels and preserving sampling semantics.
3. Use persistent request slots and bounded admission, CUDA graphs for decode,
   and batched codec windows. Evaluate cancellation and slot reuse separately.
4. Only replace a profiled kernel with Triton/CUDA when numerical validation
   and end-to-end measurement support it. KDA is a workflow reference, not a
   prevalidated Drama 3 kernel library.
5. Introduce a Rust control plane only if measured Python control cost warrants
   it. Engine language is optional in the user request.

## Quality and promotion

Use upstream Seed-TTS WER and speaker-similarity evaluation on saved audio.
Check numerical parity of each changed operator, prompt construction and
state transitions. Sampling is stochastic; identical audio bytes across frameworks
are not a valid universal quality criterion. Preserve audio for listening.

Do not promote from microbenchmarks or partial concurrency coverage. Missing
quality coverage or incomplete baselines means INCONCLUSIVE. Stop an individual
experiment on invalid audio, repeated startup failure, OOM or state corruption;
preserve the failure evidence and address its cause before continuing.

Non-goals for the first version: training, multiple GPUs, broad model support,
altered model sampling or shorter output limits. The default path retains native
precision; dense FP8 was evaluated separately and rejected for serving. Its isolated
variable was quantizing dense projections, with unchanged corpus, sampling and
FP32 recurrent state. Reject promotion if full WER/SIM coverage fails the quality
gate or the complete serving comparison does not improve. Kernel timings alone
cannot promote this path.

An independent precision experiment sets `DRAMA3_GDN_STATE_DTYPE=bfloat16`,
matching the new upstream deployment's recurrent-state storage. The default is
still FP32. Its hypothesis is lower state traffic at larger batches; dense and
expert precision, corpus, and sampling stay fixed. Validate recurrence and full
saved-audio WER/SIM before promotion. Profile the actual 2048-token graph capacity
as well as the original 512-token diagnostic so planning overhead is visible.

## Provenance

- ThinkFlowLab/OmniRSI PR 9: Strata, vLLM-Omni, SGLang-Omni knowledge packs.
- Current upstream Drama 3 model implementations and serving recipes.
- NVlabs/kda: correctness-gated, profile-driven kernel iteration.

Raw machine-local artifacts live outside the source tree; publish compact
manifests, reproducible commands, raw timing records and measured verdicts.
