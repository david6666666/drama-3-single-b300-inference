# Fixed native model / codec work: overlap and cached weights

The completed diagnostic passes **4,177 exact comparisons** across eight timed replays. The untimed recorder also reproduces the archived **80-request, 414-entry trace and all 80 request PCM streams** exactly. All 99 codec job PCM tensors, every model output and request metadata snapshot, final full model/request/codec states, and CUDA RNG match. An omitted subsystem remains in its initial state. No mismatch occurred. Exit status is zero; GPU6 is released and the source worktree remains clean.

Source is `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`, GPU UUID `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`. The immutable [declaration](experiment.json), [driver](driver.py), [support module](fixed_profile.py), [command hashes](launcher-command.json), [process audit](process-audit.jsonl), [raw result](result.json), and [CPU summary](summary.json) retain the full setup and checks. The frozen source was not modified during the experiment. The [analysis script](analyze.py) reproduces the summary from raw records.

| Fixed work | Normal codec wall, ms | Cached codec wall, ms |
|---|---:|---:|
| Model only | 1685.655 | — |
| Codec only | 176.064 | 164.218 |
| Serialized model + codec | 1861.710 | 1850.100 |
| Overlapped model + codec | 1807.996 | 1800.720 |

Cached codec reduces codec-only wall time **6.728%**, serialized wall time **0.624%**, and overlapped wall time **0.402%** in this single fixed-work sequence. These are diagnostic measurements, not replicated service speedups. One normal overlapped replay without stage events takes 1805.755 ms versus 1807.996 ms with events, a 0.124% observed difference. Both retain identical output/metadata verification copies; the no-events run isolates only event instrumentation, not all observer work.

## What the controlled sequence resolves

Normal model GPU event union rises from **1684.810 to 1805.596 ms** under codec overlap: **120.786 ms / 7.169%** longer. Decode contributes 117.960 ms of this increase; prefill contributes 2.827 ms. The same codec work's event union rises from **175.569 to 350.456 ms** under overlap. There is resource competition under an unchanged native trajectory, rather than merely different output lengths or scheduling choices.

Parallel execution still wins overall: normal overlapped wall is **53.714 ms / 2.885% shorter** than serialized. Full serialization is therefore contradicted by this fixture. With overlap, 349.165 ms of codec span overlaps model work, only 1.291 ms is exclusive, and 1.082 ms lies outside both recorded stage unions. Cached weights reduce codec union 31.487 ms, but model union only 7.219 ms and total wall only 7.276 ms. This explains why a visibly faster isolated codec can have little delivery effect.

Even the diagnostic counterfactual of eliminating all codec work exposes only **122.341 ms / 6.767%** of this fixture's overlapped wall. That is neither a serving implementation nor a universal upper bound: different schedules can also alter cache/resource behavior. It is sufficient to reject transferring the full codec-only percentage directly to end-to-end throughput.

## Scope and remaining evidence

The recipe has 80 device initializations, 13 packed prefills, 13 first-frame graphs, 184 single-step decode graphs, and 99 codec jobs. Decode runs 6,000 actual rows in 6,667 captured row slots, with mean actual batch 32.61. It includes a dynamic low-batch tail and has **no mixed graph or two-frame pipeline**. Timed replays use GPU-prestaged recorded inputs, original DP groups and immutable codec row/reset/lifetime metadata. PCM conversion and pinned D2H remain; HTTP, callback-thread delivery and the Engine's CPU output-completion loop do not.

Stage events include input copies and retained validation copies as well as graph work. Some CPU submit scopes include CUDA queue backpressure or allocation waits; they must not be interpreted as Python interpreter cost. Timed GPU work has no per-step CPU output or metadata read. The recorded outside-stage time is only about 1 ms, so there is no evidence here for a millisecond-per-step host gap.

The separate [current full-service ledger](../../current-pipeline-service-ledger.json) covers the actual mixed/pipeline configuration and finds 21.289 s model, 6.679 s codec, 5.709 s overlap, 0.970 s codec-only, and 0.613 s in neither model nor codec event span. Its overlap association is observational; the exact controlled fixture establishes competition in its own narrower setting. Neither measurement should be substituted for the other.
