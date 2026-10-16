# FC1 full-input staging: bounded CPU feasibility

Status: source-level proposal only; no new kernel, AOT, GPU or performance result. This is separate from the failed route-row cache and the slower FC1→activation→FC2 resident chain.

## Exact scope and memory arithmetic

Eight is the maximum number of routed tokens in one native token tile, not the maximum per expert. Existing fixtures include up to 64 routes for one expert. For a token tile, the original quantized FC1 input is 8×2048 E4M3 bytes (16384 B) plus 8×16 FP32 K128 scales (512 B). Original weight scales optionally need another 16×4 B per output tile. The actual M128 FC1 compiled shared footprint is 74616 B, so an additive full-input buffer gives 91512 B, or 91576 B with all sixteen weight scales. Alignment, barriers and compiler allocations must be included in a new actual launch/CUBIN report. Shared capacity alone fits the proposed roughly 112 KiB ceiling, but does not establish two resident CTAs, registers, waves, or performance.

The existing original FC1 activation operand already uses TMA gather4, with routed rows cached once per output tile in `smem_ab_resources.py:1106–1186`. Its K loop issues the gather loads at `:1205–1240`. Thus the proposal does not remove sixteen independent route-map reads from that input path: they were already cached. In the scale path, `smem_deepseek_sf_resources.py:241–291` caches tile metadata, while `:295–343` loads one K128 scale group and one weight scale per stage. The original route-map load at `:318` was the smaller route-cache experiment. Full-input staging goes beyond that experiment by changing when the input bytes and scale values are fetched.

## Minimal adapter that preserves the native consumer

1. In the LoadB task's per-work-tile prepare phase, use the original route and valid-row rules to gather the entire 8×2048 FP8 input into a dedicated shared buffer. All input bits are copied; no conversion, recomputation or quantization is introduced.
2. In the scale producer's per-work-tile prepare phase, gather all 8×16 original FP32 activation scales into its own shared buffer. Optionally gather the sixteen original weight scales for this output tile. Keep the original global expert/output-tile offsets.
3. For each of the sixteen K128 iterations, copy the appropriate input slice into the existing four-stage, 1024-byte-per-stage swizzled B pipeline. Preserve its original UMMA descriptor, MMA K32 order and accumulator/dequant sequence. Copy the original scale bits into the existing scale stage in its original order.
4. Pair the implementation with a B pipeline change from TMA producer completion to an AsyncThread producer completion, and issue the correct shared-to-async visibility fence before consumer release. The original barrier expecting a TMA byte count cannot be reused for plain STS.
5. For the SF pipeline, ordinary shared stores cannot be published solely by the existing cp.async arrival. Use an AsyncThread release with synchronous weight-scale load/store, or explicitly complete outstanding cp.async operations before the release. Keep native scale multiplication, FMA, amax/reciprocal and E4M3 output boundaries untouched.

Source call sites are `batched_gemm_tasks.py:226–309` for LoadB, including its PDL wait before prepare and stage loop, and `:1593–1651` for scale producer prepare and stage loop. The previously validated `expert-resident-full/resident_shared_b.py` supplies a useful narrow adapter pattern, but its FC2 numerical validation does not validate this new FC1 gather path.

## Ownership and limits of the minimal version

Each full buffer belongs only to its producer warp(s). Consumers continue to read the original circular stage buffers, not the full buffer. After the producer has copied all sixteen slices into acquired stages, it may reuse its full buffer for its next work tile; the original stage acquire/release contract protects the consumer copies. Any multi-warp gather must explicitly complete and publish before the first shared read. Persistent work acquisition, inactive-tile skip/drain and PDL launch placement remain unchanged.

Invalid rows must follow the original predicate/fallback without out-of-range route-map access. Global absolute mn limits must be converted to local counts exactly once. Multiple token tiles of the same expert, empty whole launches, partial rows, live routing changes and reused graph addresses are mandatory cases, not optional extensions of a single eight-row fixture.

The minimal version removes per-K dependent global supply, but retains the per-K pipeline barriers and adds a shared-to-shared copy. It front-loads the full gather and may delay early MMA issue. It does not reduce the 16 KiB input or 512 B scale payload per output tile, does not share data across output CTAs, and leaves expert weight bytes unchanged. The whole B64 input is only 128 KiB; existing L2 hit behavior must be measured, not assumed. There is no evidence that the saved dependency is a dominant service cost.

Removing all B/SF per-K barriers would require a separate full-tile publication and consumer-completion protocol: direct UMMA/epilogue readers must finish all K blocks before a producer overwrites the buffer. Bypassing the original four-stage pipeline without that epoch/lifetime proof is unsafe. That larger variant is not part of this minimal proposal.

## Smallest meaningful gate

First implement only artifact-local adapters and obtain actual AOT resource, PTX/SASS, stage and masking evidence. Then compare every original FC1 byte/scale, activation boundary, FC2 logical expanded-route output and native ordered-top8 final output across the fifteen existing fixtures, including reused graph/live metadata and synthetic empty/partial/multitile cases. Preserve original checkpoint weights/scales, native top8, native quantization boundaries, guarded outputs, immutable input/metadata/weights, RNG and graph ownership.

Only after exactness should one compare the complete native MoE call with live metadata, native activation/finalizer and PDL enabled in both arms. A full-input copy microbenchmark or FC1-only speedup is insufficient. Three real layers must each exceed the declared complete-call gain gate before expansion. No GPU execution is authorized by this CPU document. The newly proposed transparent weight compression is a more direct physical-byte hypothesis and can be assessed independently first.
