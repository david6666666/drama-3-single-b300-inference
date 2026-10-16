# Implementable native expert-resident chain

This is an implementation plan, not a speed claim. The acceptance reference remains the original native full256-expert, ordered-top8 MoE. FP8 checkpoint weights/input scales, native FC1 E4M3+FP32-scale boundary, activation E4M3+FP32-scale boundary, FC2 BF16 output, and final top8 FP32 ordered accumulation followed by one BF16 cast all remain explicit.

## Current arithmetic foundation and remaining gates

- FC1 v2: one real layer0/expert205 eight-route fixture passes42 exact checks. The full layer0 five-distribution expansion also passes valid bytes/scales, graph metadata changes and padded-output canaries. The declared synthetic all-zero input fails scales: native0 versus candidate2.23214294e-15. The first failure stopped the expansion before layers19/39. No general FC1 claim is valid yet.
- FC2 v2: helper-only BF16 shared-store logical mapping correction passes the single-expert eager/graph gate,40 checks; fifteen-oracle expansion is separate work. It has the same TMEM row0/row+8 mapping issue as FC1.
- Shared activation: teammate artifact `expert-activation-native/shared_activation.cu` has a genuine device-body ABI and CPU/AOT evidence. Its GPU exact gate is independent. Reusing the original host activation launch establishes semantics but cannot eliminate the global stage boundary by itself.

## A minimal complete work unit

One **four-CTA cluster owns one expert and one native tile of up to8 routed tokens**. Each rank `r` owns up channels `[128r,128r+128)` and gate channels `[512+128r,640+128r)`, plus disjoint FC2 output columns `[512r,512r+512)`. The expert ID and original routed-token mapping are shared by all ranks. No new routing or pruning occurs. Original all256 expert histograms determine clusters; empty work exits uniformly before any collective operation.

1. Execute two native FC1 output128 tiles per rank: up, then gate. Each runs the original16 K128 blocks, each with four `m64n8k32` MMA operations in original order, FP32 per-block scale accumulation, independent per128 output max/quantization. Write native logical FP8 bytes and scale into shared `fc1[8,256]` and `fc1_scale[2,8]`. The two matrices are independently quantized; no combined256-channel max.
2. Drain the relevant producer/MMA/epilogue pipeline and synchronize the CTA. Reuse epilogue warps0–7 for `activation_shared_8x128(fc1, fc1_scale, act_local, act_sf_local, valid_rows)`: one warp/token, lane owns four channels. It must dequantize the actual rounded FC1 FP8 values, evaluate the original activation formula/order/fast-math operations, then quantize the native128-channel activation block. Output is `act_local[8,128]` plus `act_sf_local[8]` (1056 bytes/rank).
3. All four ranks execute a **publication cluster barrier**, including inactive token lanes. Each rank gathers the other three slices into a local full `act_full[8,512]` and `act_sf_full[4,8]` (4224 bytes). This dense local form matches the native FC2 input indexing; implementing direct DSM-fed MMA would be a separate change.
4. All four ranks execute a **consumer-completion cluster barrier** after remote reads finish. Only after this barrier may a producer overwrite its local activation slice or recycle the cluster's work unit. Readers' private full copies allow FC2 to proceed independently.
5. Execute four FC2 output128 tiles per rank. The K order remains block0,1,2,3 of128, each with nativeK32 MMA subdivisions and original scale application. Store logical BF16 output into the existing globally permuted FC2 output buffer. No BF16 summation of expert partials.
6. Keep the existing native finalizer as a separate kernel initially. It loads the eight original expert branches in their original order and accumulates FP32 with original route weights, then casts once to BF16. Finalization is not part of the first resident-kernel optimization; fusing it would introduce cross-cluster ownership and must have its own exact gate.

## Reusable code and interfaces that are missing

Installed `flashinfer/prims_ts/batched_gemm/batched_gemm_tasks.py:create_epilogue_task` already separates TMEM wait/load/release from `gmem.store_epilogue`. `TmemCResource.consumer_work` and the corrected logical store/conversion can be reused as an **arithmetic interface**. `SmemAResource`, gathered input resource, `SmemDeepSeekSfAb` and `create_mma_task` provide original K128 pipeline semantics. The native FP8/SASS oracle should remain the reference throughout.

The existing task manager is not an already-composable two-GEMM resident chain. It constructs one homogeneous GEMM's work queue and resource graph. In particular:

- `GmemCResource` binds a global output pointer/TMA descriptor; it needs an artifact-local `SharedFc1OutputResource` implementing the same wait/consume/quantize contract, with explicit per-rank shared offsets and no global output dependence.
- FC1's independent output-tile work queue must become a composite `(expert,token_tile)` cluster work queue that assigns fixed rank slices. FC1 up/gate offsets and FC2 output offsets cannot independently advance persistent work IDs.
- FC2's B operand currently comes from global/TMA input. It needs a `SharedActivationBResource` that feeds the same original MMA descriptor/swizzle from local full shared activation. It must not reinterpret row-major activation as an MMA swizzle or apply the scale twice.
- The existing scheduler owns producer release and TMEM deallocation for one stage. A new composite schedule must explicitly drain FC1, preserve required shared values, reset pipeline epochs/barrier parity, then start FC2. TMEM can be reused only after every FC1 consumer has released its tile; reuse does not imply a second simultaneous accumulator allocation.
- A new cluster-lifetime resource must implement publication/completion barriers and DSM address mapping. Native MMA stays `cta_group::1`; a four-CTA data-sharing cluster is not a four-CTA MMA group.

A lower-risk first complete implementation can run FC1, shared activation, and FC2 sequentially within one composite cluster work unit, with correctness-only global debug copies of both FP8 boundaries. Eliminate debug copies only after comparing both boundaries and the BF16 result exactly. A separate multi-kernel composition gate first verifies that the independently fixed components agree when connected.

## Resource and ownership budget

Current compiled FC1 uses512 threads; dynamic register roles are:

| Warps | Role | Requested registers/thread |
|---|---|---:|
|0–7|two-warpgroup FC1 epilogue|160|
|8|MMA|48|
|9|weight load|48|
|10–11|routed input load|48|
|12|combined scale load|48|
|13|persistent work ID|48|
|14–15|padding tasks|48|

These source role hints sum to a hypothetical53,248 registers/CTA, but the emitted full-chain SASS has no SETMAXNREG, highest R29, and CUBIN REG32. The hint sum is not actual allocation and does not establish a one-CTA residency constraint. Actual occupancy is unmeasured; component exactness remains an arithmetic foundation, not a performance result.

Only two of eight non-epilogue warps are padding; deleting them changes the source-hint sum to50176, which still does not predict actual allocation. The source forces eight epilogue warps for the two `m64` halves and256-thread max reduction. Reducing to four epilogue warps therefore requires explicit serial consumption of the two TMEM halves plus a changed max-reduction schedule; simply overriding the warp count is incorrect. It is also not justified to lower `setmaxnreg` based on the static header: each task's live ranges and generated code need inspection. Combining persistent scheduling with an existing producer warp requires changing the producer/work-queue handshake, not only an index assignment.

FC1's measured shared allocation is73592 bytes; FC2's is74664 bytes. Additional persistent state can be bounded using phase lifetimes: `fc1+scales`2112 bytes is dead after activation, while `act_local+full_gather`5280 bytes is live during DSM transfer. Reusing the former storage yields a conservative maximum `74664+5280=79944` bytes, leaving1976 bytes under80KiB for alignment and new synchronization state. This is only a planning bound; the composite allocator/AOT must report actual aligned bytes. Keeping all buffers simultaneously instead yields82056 bytes and already violates80KiB. Reusing original epilogue scratch may save more but must be proven safe across producer epochs.

Each cluster still reads the same original3MiB of weights per expert/token8 group. Two FC1 tiles plus four FC2 tiles per rank preserve192 weight TMA tiles/cluster. Fusion reduces global barriers/intermediates and producer parallelism at the same time, so bandwidth can get worse. Prior three-layer NCU/old profile figures establish coverage, not expected speed. The retained actual native intermediate oracles make failed math diagnosable before attempting performance.

## Promotion order

1. Finish generalized FC1/FC2/shared-activation exact gates, including signed zero, padding and empty routing. Preserve every failure.
2. Connect one expert's stages with native boundaries and native ordered finalization; validate eager and retained CUDA Graph storage ownership.
3. Compile the minimal four-CTA composite kernel with actual shared/register/TMEM/cluster limits and no spill. Re-run the same boundary checks; then multiple experts and changing histograms using all15 immutable native oracles.
4. Only after exact passes measure **complete MoE**, including routing, stage synchronization, output writes and finalization, against the original native fullcall. At least5% local net improvement is required before broader runtime validation; this local gate does not replace the user's all-concurrency1.5× service goal or WER/SIM gates.


## Implemented bounded composition update

The current actual outer uses four physical CTAs, with original group-1 MMA in each. Each rank computes FC1 output tile rank then rank+4 (its up128/gate128). After exact native activation and ordered DSM publication/read-completion barriers, each rank computes four FC2 tiles rank+4*phase. These are disjoint interleaved 128-column pieces totaling 512 output columns per rank; they preserve the native ordered K128 accumulation.

The outer allocates TMEM32 once, relinquishes allocation permission once, and lends the same base to all six fully drained native stage invocations. No intermediate stage may allocate or deallocate TMEM. The outer deallocates once after the final stage and full-CTA convergence. This avoids illegal allocation after relinquishment. Each stage asserts its required TMEM capacity is at most 32 columns. Native dynamic role-register changes are retained at every stage boundary.

The explicit outer shared allocation is 78,080 bytes aligned to 1024, including native scratch [0:72704), FC1 [72704:74816), local activation [76928:77984), and outer TMEM-pointer storage at78016. Full activation [72704:76928) aliases FC1 only after every activation consumer has finished. Full-chain actual AOT resource and exact results must replace component-only estimates before any promotion.


Register-accounting correction: the 160/48 PTX/source role hints are not equivalent to actual SASS allocation. For the full debug and zero-debug resident CUBINs, cuobjdump reports REG32, emitted SASS has no SETMAXNREG, and the highest general register is R29. Any earlier 53,248-register/CTA or one-CTA-per-SM claim is an unproven source-budget inference and must not be used as occupancy evidence. Shared-memory totals and zero-spill checks remain valid; actual occupancy requires its own measured/compiler evidence. See resident-register-accounting-erratum.json.
