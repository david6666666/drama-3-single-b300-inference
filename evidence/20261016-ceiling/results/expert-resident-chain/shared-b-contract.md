# Resident FC2 shared operand seam

Status: CPU/AOT passed, **not GPU validated or performance measured**. `resident_shared_b.py` retains native SmemBResource/SmemDeepSeekSfAbResource consumers and all native FP8-to-FP32 dequant/accumulation. It changes data sourcing and producer publication, not arithmetic.

## Integration

- Import `ResidentSmemBResource`, `ResidentDeepSeekSfAbResource`, `resident_pipeline_configs`.
- Only for resident FC2: replace `pcfgs` with `resident_pipeline_configs(cfg, pcfgs)` **and** instantiate both adapters. This pair is mandatory; retaining TMA transaction-byte expectations for synchronous stores deadlocks/violates publication.
- B resource receives all original kwargs plus `shared_act`: caller-owned `cutlass.Array`, shared address space, Uint8 flat `[8*512]`, logical row-major E4M3 bytes. SF resource receives all original kwargs plus `shared_sf`: shared Float32 flat `[4*8]`, K128-block-major then token. Both references are explicit; no mutable module/global state or pinned offsets.
- The full tensors must represent the current token tile. Native `coord_b_mn`/`mn_limit` determine valid rows; the B adapter uses local rows 0..7 in that pinned tile. Native SF metadata still determines weight expert/output block and valid token limit. Neither adapter can independently switch the resident tile.
- Keep cfg cluster_m=1, tile_n=8, tile_k=128, E4M3, native MMA64n8k32. LoadB is native warp8, LoadSfAb native warp12, one warp each. Four-CTA outer DSM grouping must not change native MMA group or per-stage pipeline cluster layout.

## Memory and synchronization

One warp copies two 16-byte chunks per lane, 1024 bytes/K128, to original B stage via `(row*128+channel) XOR (row*16)`. Invalid rows zero-fill. The original descriptor (leading/stride1024, layout2) and stage depth are inherited. The caller retains the shared source until all FC2 output tiles drain. Each lane fences STS for UMMA's async proxy; original task commit publishes through AsyncThread-to-UMMA mbarrier, 32 producer arrivals, one native UMMA consumer. This is an actual LDS.128/STS.128 staging body, not a global tensor alias.

Scale staging remains native 32 FP32 activation slots plus one FP32 weight slot. Eight or fewer scales are read from shared memory, other slots zeroed. The one weight FP32 is synchronously loaded from the original global offset and stored unchanged. Both SF sources now publish via AsyncThread-to-AsyncThread; original epilogue wait/release and K-block dequant/FMA order are inherited. Avoid mixing ordinary shared STS with the original cp.async-only completion barrier. No internal full CTA barrier, DSM access, scratch allocation, TMEM allocation/relinquish/deallocation, or altered native scale math.

Outer lifecycle remains: FC1 epilogue drain and CTA sync → activation helper → CTA publication → cluster publish → gather all four fragments → cluster read-complete → resident FC2. Gather source must not be reused before all ranks complete reads. The caller is responsible for fixed worktile, stage allocator/barrier lifetime, and the final drain before shared reuse.

## Evidence and limits

`resident_shared_b_aot.py` compiled real staging bodies for sm103a without initializing CUDA; 36 CPU cases cover row counts0..8 × four K128 blocks, bijective complete s128b destination ownership, and bounds. `shared-b-aot/kernel.sass` shows LDS.128/STS.128, MEMBAR.ALL.CTA and FENCE.VIEW.ASYNC.S; SF is native FP32 LDS/LDG/STS. The first offline binary-extraction regex accidentally selected a host metadata string after successful code generation; `shared-b-aot-extraction-failure.json` retains that infrastructure error, corrected to the existing exact @kernels_binary parser. No GPU ran.

Full adapter scheduling/composite compilation, actual UMMA operand interpretation, partial row/guard checks, pipeline stage reuse, outer DSM barriers, and all native FC1/activation/FC2 boundary bytes still require the coordinated chain gate. Standalone activation DSL independently passed137 cases/2066 checks in `../expert-activation-dsl/result.json`; it does not imply composite correctness or speed.
