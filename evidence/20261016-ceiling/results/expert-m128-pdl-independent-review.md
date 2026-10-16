# Independent CPU review: M128 PDL

Reviewed the candidate and gate without editing their files or running a GPU. No blocking source-level PDL cycle or ownership flaw was found. This is a CPU review, not a proof that the kernels terminate or retain exact output on hardware; the declared exact/empty/graph gates remain required before timing.

## Important correction: this is the persistent path

The global M128 component configurations use `tile_scheduler=1`, `cluster_m=1`, `use_early_exit=1`, `use_clc_fast_drain=0`; the new candidate changes both PDL flags to 1. Earlier reasoning based on the static outer-CTA return does **not** apply. In this candidate a zero-active-work launch still initializes resources/TMEM and runs the persistent CLC skip/drain path. It is not a zero-cost launch or an immediate return.

`batched_gemm_kernel.py:2417` waits before reading dynamic metadata. In the persistent body, lines 1786–1815 cache `num_non_exiting_ctas` and build `BatchedGemmWorkQueue`. `smem_misc_resources.py:141–169` skips each work tile whose token-CTA index is at least that count. With count zero, every payload is skipped. `batched_gemm_tasks.py:69–110` keeps wait/advance/release outside the skippable payload, and the WorkId task at 1773–1785 keeps acquire/fetch/commit/wait/advance/release outside it as well. All participating tasks therefore still advance the same CLC lifecycle. Setup, final pipeline drain and TMEM deallocation still occur.

The PDL launch actions are after the whole work loop in Gather/LoadB and scale-load tasks, so they are reachable after an empty work queue drains. Actual AOT PTX agrees: FC1 has entry WAIT at line 79 and launch_dependents at 1293/2155; FC2 has entry WAIT at 74 and launch_dependents at 1325/2135. These are not launches inside a skipped payload. CUDA's installed `cuda_device_runtime_api.h:435–465` additionally specifies that all CTAs exiting implicitly releases dependent scheduling, and that WAIT blocks for direct predecessor completion. This supports an acyclic chain; it is not a formal proof of every generated barrier state.

## Full native chain and numerical checker

The candidate sequence is FC1 → original activation → FC2 → original finalizer, with PDL enabled at all four launches. Native activation triggers launch then waits before reading payload; finalizer waits before reading routed output. The baseline remains production tactic `(8,5)` with PDL enabled. Input quantization and logits-to-top8 are outside both complete prepared-body timings. Both arms include live original routing metadata, all body kernels and finalization in the timed call.

Each arm owns separate metadata/workspace buffers. `check_metadata` checks route-map bijection/range, expert and token ownership, padded count, active-row prefix and each local valid count. `check_outputs` uses the candidate's own live map to recover original expanded-route order; the vendor arm uses its own map. This permits legal physical packing changes without sorting by output value or weakening bit-exact FP8/scale/BF16 checks. All 15 archived cases, guards/tails, input/weight/RNG ownership and reused graphs remain in the gate.

Safe handoff sources are the unchanged `expert-m128-body-v3/probe.py` checker and `expert-m128-body-timing/timing.py` timing design. The new driver incorporates their independent metadata fix. The declared continuation rule remains at least 5% complete-call reduction in each of three real-case medians, no greater than 5% regression in any of 12 stress cases. One-round feasibility is a screen, not the final timing claim.

## Empty test and graph evidence

`pdl_gate.py` keeps valid buffers/descriptors and a fixed nonzero launch grid, sets only live ncta/padded to zero, and calls FC1 → FC2 only. It does not repopulate metadata, run activation or run finalizer in that synthetic domain. Eager and retained-graph replays check all output sentinels/guards, metadata, inputs and RNG. Before returning to positive work it restores the original outputs, including native activation bytes/scales, so the two-GEMM graph's FC2 input is valid despite the absence of activation in that graph. It then rechecks both the same two-GEMM graph and the full four-kernel graph. Captured descriptor/function/tensor owners remain alive in the enclosing scope.

The queried four-kernel graph and two-kernel graph are unmodified and retained through instantiation/replay. Local cuda-python documentation confirms the GetNodes/GetEdges return tuple and typed edge-data API used by the gate. The gate requires at least three/one programmatic edges respectively and records endpoints, ports, kernel names and attributes. An unsupported legacy node-attribute query is explicitly recorded; typed dependency edges supply the primary evidence. This is launch/dependency evidence, not a measurement of useful kernel overlap. Graph replays are serialized; this does not authorize concurrent replays over the same mutable buffers.

## Scope and remaining hardware checks

The inspected launch watchdog hashes inputs before launch, refuses old result files and pre-existing GPU owners, kills only its child process group on timeout or a foreign process, and stops on any exact-gate failure. GPU2 execution belongs to the coordinator, not this review. The entry WAIT precedes GEMM payload/prefetch, so PDL should not be assumed to create broad weight-load overlap or a large service gain. No new performance result, service claim, native-contract relaxation, runtime edit or GPU invocation was made in this review.
