# CPU follow-up: one native routing pass for partitioned bodies

No additional GPU job or source edit. The first feasibility remains 135 exact PASS / performance FAIL. There is no component profiler trace for it: only complete CUDA-event times exist, so no measured microseconds can be assigned to metadata, copies, grid exits or GEMM overlap.

## What the first prototype really added

`driver.py:partition_func` recomputes input quantization, then copies q/scales/logits into addresses referenced by its persistent runner inputs. These copies total 200704 payload bytes per layer (131072+4096+65536), or 401408 logical read+write bytes. It then performs one complete native canonical router, one int16→int32 route-ID copy, two precomputed-routing metadata builds, two activation launches and one ordered finalizer. The two partition GEMM static bounds total352 token tiles instead of native288; useful token tiles are unchanged. The complete serial overhead versus native was32.102/33.485/34.406µs for layers0/19/39; parallel execution recovered9.779/12.800/12.802µs, leaving22.323/20.685/21.605µs net losses. These are arithmetic differences of measured whole graphs, not attribution to individual kernels.

A direct producer can eliminate the three copies while preserving ownership: call the existing unchanged `drama3.moe_input.prepare_kernel` with preallocated q/scales/logits that are already bound into each runner. Do not merely remove the copies while continuing to allocate different producer outputs; that would leave bodies reading stale addresses. Apply direct producer binding identically to control and candidate in any routing-only comparison. No performance estimate is justified until measured.

## Why slicing global metadata is not enough

Native routing writes experts in increasing expert order. `RoutingKernel.cuh:710–770` computes a prefix sum of token8 group counts, writes expert-relative batch IDs, and writes CTA `MnLimit` using padded global token offsets. `expanded_idx_to_permuted_idx` and `permuted_idx_to_token_idx` use the same padded domain. For split128, the boundary `b=sum_{e<128}ceil(count[e]/8)` varies on the GPU every replay.

A pointer view at expert128 is unsafe by itself. The second body needs local expert IDs, local CTA/row coordinates and local padded-stride scales. The native FC1/activation/FC2 kernels compute FP8 scale addresses with `actual_total_padded_tokens`, not allocated capacity. Reusing the full global scalar with local compact bytes, or subtracting the prefix only from one map, silently changes scale/activation addresses. A host `.item()` to obtain a dynamic view would also add a service synchronization and break the desired graph path.

A **safe small metadata transform** is possible without rerouting or copying full activations. After one native global router, a GPU kernel derives `b` by counting valid global CTA batch IDs below128 (mask by `num_non_exiting_ctas`) and creates two compact metadata records. Do not read `num_tokens_per_expert` merely because it is allocated: the custom routing runner accepts that parameter but does not forward it in the inspected method4 branch, so that scratch may be unwritten. The CTA maps and non-exiting count are the authoritative initialized inputs:

- Partition0 has `base=0`, tile count `b`; partition1 has `base=8*b`, tile count `global_count-b`.
- For each local CTA i, copy native global CTA `i+base/8`, subtract partition expert offset from its batch ID and subtract `base` from its MnLimit.
- For each expanded route, if its native global expert ID belongs to the partition, set local map=`global_map-base`; otherwise −1. Preserve original k order and native BF16 weights.
- Copy local permuted→token indices from the global prefix/suffix, preserving native intra-expert order; retain the required +1 guard element for vendor over-read.
- Write local total padded tokens and non-exiting CTA counts. Handle empty partition explicitly, with initialized inactive/canary storage and no invalid pointer arithmetic.

Each body then owns a fixed, disjoint slice of FC1/activation/FC2 buffers. They may come from one allocation, but that is **one allocation with two local layouts**, not a single global-stride activation layout. No model arithmetic or precision changes; compare all effective native bytes/scales/FC2 against the existing oracle.

This replaces two full precomputed-routing builds with one compact descriptor transform. It does not by itself reduce the static176+176 tile grid, remove the second activation launch, or prove that the remaining body pipeline will beat native.

## Smallest native launcher seam

Existing prepared-body ABI already takes caller-owned metadata and7 body buffers and invokes the original vendor FC1, native activation and vendor FC2. It is ideal except that `data/csrc/trtllm_fused_moe_kernel_launcher.cu:4296` and`:4325` reject `do_finalize=False`. Internally `MoE::Runner::run` already guards finalization with `args.do_finalize` (`trtllm_fused_moe_runner.cu:780`). A minimal artifact-local C++ variant can permit no-finalize prepared DeepSeek bodies and return/retain caller-owned FC2 without adding any numeric kernel. Keep the installed reference module untouched and retain the old exact result as oracle.

Build approach: use an independently named JIT module with only a copied launcher translation unit adjusted at the two prepared-body guards; all vendor GEMM, native activation and routing sources remain the installed originals, same compile flags and cubin libraries. Verify APIs retain input/output widths, workspace lifetimes, original tactic `[8,5]` and `args.do_finalize=False`. Do not enable DA distribution policy or share mutually-live max-workspaces between streams. The existing prepared record binds `workspace.ProjUpTileN=8` and independent total-padded scalars. Both partitions retain separate body scratch/workspace, even if their immutable weights and input tensors are shared.

An existing lower-level native finalizer is exported as `runtime.trtllm_moe_run_finalize` (`core.py:4078`, launcher`:3051`); this could replace the custom finalizer only after combining FC2 buffers/map into one coherent layout. That would add gathers/copies or a new consumer mapping, so the already-exact two-map finalizer is the simpler fixed control for this prototype.

## Can the lower-level launch avoid the +22.2% grid overhead?

`TrtllmGenBatchedGemmRunner::run` exposes a host `maxNumCtasInBatchDim` argument (`batched_gemm/KernelRunner.h:128`), and maps it to `mMaxNumCtasInTokenDim` (`trtllm_batched_gemm_runner.cu:364`). The standard `PermuteGemm1::Runner` and`Gemm2::Runner` recompute this bound internally from numTokens/topK/localExperts; there is no public graph-safe dynamic per-partition grid size. Reducing both halves to144 is invalid: one 128-expert half can legitimately require176 tiles, even if these three observed inputs are smaller. Changing numTokens/topK to force a smaller launch would change input strides and route semantics.

A custom low-level wrapper can select a smaller **proven bound** for a fixed captured routing distribution, but that would be a fixed-input-only micro result until a safe runtime fallback/conditional graph selection is built. It must not be presented as service-ready. Fixed ranges of global CTA rows could sum to288, but require row/base/scale-layout rebasing and can split highly unequal useful work; this changes the partition strategy and is a separate experiment. Current CPU review does not establish a zero-copy global-layout solution through the public wrapper.

Installed `flashinfer/prims_ts/moe/runner.py:2230–2384` also exposes staged DeepSeek FP8 orchestration with explicit GEMM IO and native activation. However it compiles Prims-TS GEMM via `get_compiled_gemm`, not the same frozen vendor cubin. It is valuable for a later editable kernel path, but is not automatically the unchanged-native-GEMM candidate. Require FC1/activation bytes/scales and FC2 exact gates before calling it equivalent; do not silently substitute it just because config mapping exists.

## Concrete next single-variable feasibility, pending authorization

Freeze a new artifact module/harness. First compare native prepared `do_finalize=True` against the original to validate the module build. Then validate the no-finalize prepared body against original per-route FC2 and the already-exact finalizer. Use the same fixed2×128 split, same native tactic/PDL, original weights and disjoint buffers.

For the routing experiment, both arms use direct stable input preparation and the same no-finalize body wrapper. Control populates local metadata through existing native precomputed-routing paths; candidate fills it from the single native global router using the one rebase kernel. Thus the only variable is local metadata construction. Retain serial/parallel comparison, include global routing and the transform in total layer time, and separately compare the best complete partition graph against original native MoE. Preserve all successful reference intermediates as CPU blobs this time.

Use the same real three layers plus changed expert-count distributions and empty-half inputs for correctness. Global native routes must remain authoritative; do not compute top8 on CPU or renormalize local weights. Record all maps, actual padded counts, canaries, pointer ownership and untouched inputs, weights/RNG. Exact failures stop promotion; no timing until gates. Require ≥5% complete local MoE improvement against original native before broader validation. If metadata reuse alone cannot recover the observed20–22µs gap, retain its evidence and proceed to a genuinely different producer/consumer structure rather than sweeping splits.
