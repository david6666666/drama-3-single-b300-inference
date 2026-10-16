# Metadata-ready FC2 PDL specialization — CPU stage only

This candidate changes when the M128 FC2 weight producer may begin. It preserves the native full 256-expert, top-8 computation, all checkpoint FP8 bytes/scales, each FC1/activation quantization boundary, BF16 FC2 output, original ordered finalization, persistent work scheduling, and early-exit handling. It has not run on a GPU and has no performance result.

The earlier `expert-m128-pdl` experiment remains immutable: 1,498 exact checks passed, but a single three-real-layer screening missed the requirement that every real layer reduce complete MoE latency by at least 5%. Its twelve stress distributions were accuracy-tested; their performance was NOT_RUN. This candidate does not reinterpret that result.

## Why a separate contract is necessary

The ordinary FlashInfer validator correctly rejects PDL plus early exit without the initial metadata wait. `metadata_ready_config.py` introduces a separate typed configuration, accepts only the fixed native FC2 configuration and B64/top8/256 IO extent, and delegates all ordinary checks to the installed validator on a separate equivalent wait-enabled object. Only its native derived prefetch-depth field is copied back. Every native configuration field is pinned to its canonical value, allowing only the exact pre/post `compute_warp_layout` values of derived fields. An initial narrower whitelist accepted optional work-queue stages and TMEM-barrier changes; that CPU-only version is preserved in `cpu-initial-config-scope/`, and new rejection cases close those gaps. The installed validator/source remains unchanged and still rejects the raw unsafe configuration.

The compile-time field is not a runtime readiness proof. The public host wrapper must capture without instantiating, verify every metadata producer, add their DEFAULT full-memory edges to FC2, preserve the activation→FC2 PROGRAMMATIC edge, then seal before instantiation/replay. Plain eager launches require metadata completion before launching the computation chain. No caller may infer readiness from a programmatic trigger alone. NVIDIA documents that triggers permit scheduling while the secondary dependency wait supplies the required visibility. [CUDA PDL documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

```text
all routing-metadata producer nodes ── DEFAULT / complete + memory visible ──► FC2
                 │                                                    ┌────┴─────┐
                 └──► FC1 (original global WAIT) ──► native activation │ weight A │
                                                    │                │ may load │
                                                    └─ PROGRAMMATIC ─┤ B/SF WAIT│
                                                                     └────┬─────┘
                                                               original finalize
```

The installed original `activationDeepSeekKernel` (`trtllm_fused_moe_dev_kernel.cu:247–252`) triggers programmatic dependents before its own grid-dependency wait. Thus weight prefetch may begin while activation still waits for FC1; the FC2 B/SF waits continue to protect actual activation consumption.

The extra full edges do not serialize FC2 behind FC1 or activation. They synchronize the independent routing/count tables. Actual overlap remains opportunistic and may have no benefit.

## Actual CPU compilation evidence

`prepare.json` is CPU_AOT_PASS with 26 checks; CUDA remained uninitialized. The final stricter configuration recompiled to the same CUBIN SHA-256 `ea52321ec64fc9976a306b7c7aeeb0805033c2e024034e5cc6ce266e7d95b170`. `cpu-wait-placement.json` records the exact compiled binary and instructions. The FC2 kernel has two task-local WAITs and two release sites. In the generated PTX, LoadA warp 5 issues weight TMA before either wait. LoadB warp 6 waits before activation TMA; LoadSfAb warp 8 waits before activation-scale consumption. SASS confirms corresponding weight UTMALDG and two later ACQBULK sites. The source retains the original early-exit skip/drain path, so inactive tiles cannot be treated as valid CAP rows.

Actual resource accounting is 72,616 dynamic plus 2,048 static shared bytes = 74,664 bytes/CTA, zero stack/local/spill. The header has REG32; no actual SETMAXNREG instruction occurs in this CUBIN. PTX register hints do not prove dynamic residency or occupancy.

A first CPU host-check failure confused the last two IO shape fields. It is preserved in `cpu-initial-shape-order-failure/`; the checked original tuple is `(M,N,K,experts,tokens) = (2048,2304,512,256,64)`. No GPU or numerical result existed at that failure. A separate CPU-only failure while tightening the derived prefetch field is retained in `cpu-derived-prefetch-default-failure/`: the generic default is 3 but this exact non-gather path derives 0, so the factory now explicitly fixes 0 and rejects 3. This removes any dependence on a prior concrete-shape validation call.

## Required entry and future GPU gates

The host wrapper must reject incomplete/changed captures, unknown producer or consumer identities, missing full metadata edges, a missing or changed activation programmatic edge, use with an unrelated graph/generation, and replay before seal. It must retain descriptors, all metadata/input/weight tensors, compiled functions, graph and workspaces. Eager permits are scoped to a completed metadata event and a single launch sequence; they cannot authorize graph replay.

After CPU review the coordinator authorized only the frozen exact gate and one three-real-case screening on GPU2. Formal repetitions require a follow-up decision. The exact gate must compare all fifteen original native oracles using each arm's own live routing maps, prove original-token/expert ownership, check all five boundaries/final output, immutable state/RNG/guards, and change routing metadata within the same sealed graph. A separate zero-count graph includes actual zero-count producer nodes and full edges, fixed nonzero GEMM grids, no activation/finalizer, unchanged outputs, and positive-state restoration using the same graph. Any mismatch stops the experiment and is preserved.

Only after exact/empty gates may one screening measure three real layers against the unchanged native `(8,5), PDL=True` baseline. It must include live metadata generation, full MoE arithmetic/finalization, and the new graph edges. Three AB/BA/AB repetitions over all fifteen cases require every real layer ≥5% reduction in screening. Formal continuation requires each real layer ≥5% reduction and every pressure case ≤5% regression; unevaluated performance fields are null/NOT_RUN. Common input quantization and logits top-8 routing are excluded equally, and no local timing is a service gain.

Final integration uses metadata_driver.py and metadata_gate.py. The graph guard passed 44 CPU checks and independent review. A repeated CPU compile initially reused an existing artifact dump directory and deliberately failed its one-new-CUBIN assertion; that report/source/dump are preserved under cpu-driver-repeat-failure. Compilation was repeated into a fresh task-owned dump directory without changing installed or shared caches.
