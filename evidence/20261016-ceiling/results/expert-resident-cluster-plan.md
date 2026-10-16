# Native expert-cluster fusion: CPU feasibility and the next bounded gate

Status: **CPU design only; no new GPU launch, production change, performance claim, or promotion.** The two expert-partition experiments remain exactness PASS / performance FAIL. The latest shared-routing experiment passed 1,373 checks, but its complete-MoE medians remain 13.662 / 12.904 / 13.672 µs slower than the original native call in layers 0 / 19 / 39. Sharing metadata saved only 2.414 / 2.814 / 3.019 µs versus the otherwise identical split control. These totals do not identify a particular remaining kernel's cost.

**Recommendation:** the next useful arithmetic gate is one native expert's FC1 output bytes/scales, followed by a four-CTA producer/consumer cluster. This changes the lifetime and scheduling of the FC1→activation→FC2 body, rather than another N-tile or expert-partition sweep. It is implementable in capacity terms, but exactness and sustained bandwidth are both unproven. A new device kernel is necessary: a C++ launcher cannot call the existing vendor kernel bodies from inside a cluster.

Reproducible CPU ledger: [audit.py](expert-resident-cpu/audit.py), [audit.json](expert-resident-cpu/audit.json). The script reads archived tensors with `map_location='cpu'`, records source/oracle hashes, and finishes with `cuda_initialized=false`. Framework reference is frozen `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`. Original numerical cubins and installation remain unchanged. All throughput and quality targets are unchanged; global status remains **INCONCLUSIVE**.

## 1. The arithmetic that must survive fusion

Dimensions are H=2048, I=512, 256 experts and original top8. Input and expert weights retain original E4M3 bytes and FP32 block scales. Native FC1 stores **E4M3 bytes plus FP32 scales**, activation stores a second independent E4M3/FP32 boundary, and FC2 stores BF16. The finalizer accumulates all original eight routes in their original order before one BF16 cast. A BF16 FC1 intermediate, MXFP8 scale substitution, partial BF16 expert-group sum, split-K tree reduction or discarded expert is not this experiment.

Archived NCU kernel names identify `t128x8x128...m64x8x32`, including FC2. Thus the proposed implementation keeps **MMA m64×n8×k32**, two M64 subtiles for each output128 tile, and four K32 operations for each original K128 scaling block. It must not silently replace these with M128 or one long K2048 accumulation.

Installed FlashInfer's source provides an implementation starting point, but it is not proof that a newly compiled kernel matches the frozen vendor cubin:

- `prims_ts/batched_gemm/tmem_c_resources.py:455`: DeepSeek resets the tensor-core accumulator at the first MMA of each K tile; subsequent K32 contributions accumulate inside that block.
- `smem_deepseek_sf_resources.py:414` and `:447`: multiply the input/weight FP32 scales, then apply each block partial to the running FP32 value using packed RN FMA with FTZ. FC1 has 16 ordered K128 blocks; FC2 has four.
- `gmem_c_resources.py:1617`: FC1 output quantizer computes its quantization multiplier with an approximate reciprocal of `max(amax,1e-12)`, and stores `amax*(1/448)` as the dequantization scale. This is **not the same recipe** as the following activation quantizer. Reusing an existing generic per-token quantizer without checking would be an arithmetic change.
- Native `data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:236–374`: dequantize FC1 up/gate separately, evaluate original gated SiLU, reduce absolute maximum over each128-channel block, use `max(amax/448,FLT_MIN)`, and cast the division to E4M3. Installed native `build.ninja:43` enables `-use_fast_math`; copying the source with different exp/division flags is not automatically equivalent.

The FC1 recipe above describes the installed Prims-TS implementation. Its equivalence to the actual reference cubin remains the first GPU question. Use native bytes/scales as the acceptance reference, not Prims-TS output as a replacement reference. The byte oracles do not expose pre-quantization FP32 accumulators, so a mismatch can be localized to FC1 but may need a separate diagnostic to distinguish reduction versus epilogue differences.

Official ISA permits single-CTA F8F6F4 M64/N8/K32, with FP32 accumulators; a four-block cluster does not imply a `cta_group::4` MMA. Nonexclusive TMEM allocation starts at32 columns across128 lanes (16 KiB), so the logical8-KiB accumulator estimate is not the actual allocation. The design uses independent `cta_group::1` MMA and cluster communication for activation. [NVIDIA PTX ISA, matrix shapes and TMEM allocation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions).

## 2. A concrete four-CTA dataflow

One cluster owns one expert and one native token8 group. Rank r=0..3 owns FC1 up channels `[128r,128r+128)` and corresponding gate channels `[512+128r,512+128r+128)`. It computes these using the original M64/K32 shape and K128 scale order. Each rank has disjoint weight rows; nobody recomputes the complete FC1.

After both FC1 halves have passed through their original FP8 quantization boundary, each rank computes its128 activation channels and stores E4M3 values plus scales. Those four fragments together are the complete512-channel activation. Publish fragments, synchronize the cluster, copy the other three fragments to each CTA's local shared buffer, then synchronize again before any source buffer is reused. The second synchronization is about ownership, not arithmetic.

Each rank owns a disjoint512-channel FC2 output range. It streams four output128 tiles, each formed from two M64 subtiles. For each tile, consume activation K128 chunks in order0,1,2,3, with the original per-block scale application; cast to BF16 only after all four contributions. The original global ordered top8 finalizer remains outside this prototype. There is no FP32 atomic or inter-CTA FC2 reduction.

DSM cannot be treated as arbitrary remote memory after a producer exits. All blocks must be resident before communication and stay alive until remote accesses complete. Shared-memory limits are per block; cluster occupancy must be queried for the compiled kernel. The planned local replica also avoids relying on whether a particular MMA operand descriptor can directly consume a remote CTA's shared address. [CUDA13.0 programming guide, distributed shared memory](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#distributed-shared-memory).

The implementation must distinguish cluster synchronization from the fences and mbarrier completion needed for TMA and shared-to-MMA visibility. Double-buffer ownership is released only after the corresponding MMA reader is done. Invalid token lanes participate in synchronization and have neutral arithmetic; only their final stores are masked. No barrier can be skipped by an early-exiting padding lane or expert-empty CTA.

This first version does **not** overlap FC2 of an expert with unfinished FC1 chunks of the same expert: all four activation chunks are gathered before FC2. It removes the global all-expert stage boundary, allowing different clusters to progress independently. Native PDL already overlaps some phases, so the amount of newly exposed overlap is not known in advance. Keeping FC2's original K order is simpler and safer than a chunk-ready protocol.

## 3. Working set and bandwidth risk

The full expert's3 MiB weights cannot reside on chip. Only the short-lived activation and accumulators are retained; weights still stream. For one token8 group:

| Resource | Per CTA |
|---|---:|
| FC1 paired up/gate128 weights, K128, two stages | 65,536 B |
| Input token8×K128, two stages | 2,048 B |
| FC1 E4M3 bytes and FP32 scale scratch | 2,112 B |
| Complete activation512 plus four scales per token | 4,224 B |
| Logical shared subtotal | 73,920 B |
| Additional scale staging estimate | 288 B |
| First compile shared-memory budget, including alignment/barriers | 80 KiB |
| FC1 persistent FP32 values | 8,192 B, equivalent to16 registers per128 consumer threads |
| FC2 persistent FP32 values for output128 | 4,096 B |
| Minimum nonexclusive TMEM allocation | 16 KiB |

The 80-KiB figure is a design budget, **not a compiler result**. FC1 and FC2 staging storage can be reused after all readers finish. Actual register allocations, spill bytes, TMEM stages and cluster placement remain mandatory measurements. With proposed256 threads, two CTAs/SM require at most128 registers/thread and shared memory below roughly116,736B/CTA before allocation details. Both are capacity conditions, not an occupancy guarantee.

A minimal numerical implementation may initially use128 threads and sequential producer/consumer phases, allowing the exact128-thread activation reduction to participate across the whole block. A later256-thread role-specialized version needs an explicit subgroup/barrier implementation: blindly inserting `cub::BlockReduce<128>` into a256-thread CTA is unsafe. That numerical bring-up is not eligible for a speed claim.

Replication costs12,672 DSM bytes per cluster (four ranks each copy three1,056-byte fragments). Across the actual three layer route matrices this is2.64 /2.13 /1.85MB of on-chip traffic. It is small compared with weights, but it is an added cost; DSM latency and cluster barriers may matter even when bytes are small.

The TMA accounting exposes the principal risk. Each candidate CTA streams32 FC1 weight tiles and16 FC2 weight tiles of16 KiB; four CTAs still issue192 such tiles per expert/token8. Native does the same total work across8 FC1 and16 FC2 output CTAs. We remove neither weight bytes nor TMA transactions. We reduce simultaneous producer CTAs and lengthen each CTA. A two-stage implementation must sustain enough in-flight requests, potentially with two resident CTAs/SM, to compete with the native eight-stage,149,424-B-shared implementation. If it cannot, fusion makes bandwidth worse.

| Actual FI layer,63 real +1 padding | Active experts | Token8 groups | Four-CTA total | Waves at148 /296 CTA capacity |
|---|---:|---:|---:|---:|
| 0 | 206 | 208 | 832 | 5.62 /2.81 |
| 19 | 159 | 168 | 672 | 4.54 /2.27 |
| 39 | 115 | 146 | 584 | 3.95 /1.97 |

These are arithmetic wave counts, not cluster occupancy predictions. One CTA per expert would expose too little parallelism in layer39. Four CTAs per token group give more work, but synchronization, GPC placement and short final waves can still leave SMs idle. A fixed block-index mapping to the existing288-capacity canonical metadata avoids a device-to-host count read; unused clusters must exit uniformly before acquiring resources or crossing barriers. Every real route still appears exactly once.

Experts with more than8 routes create several clusters. Relative to one weight request per active expert, these groups request0.97%,5.66%,26.96% more logical weight bytes in the three layers. Native NCU already observes DRAM reads close to one active-expert footprint, so cache reuse is handling much of this. Spreading clusters far apart can lose that reuse. Persisting all token groups inside one expert cluster could reduce L2 requests, but would reduce parallelism and increase live accumulators; it is a separate later design, not an unrecorded change to this first prototype.

## 4. Covered time and conditional best case

Deleting FC1/activation global writes and reads alone is insufficient: their padded logical volume is5.27 /4.26 /3.70MB, just0.81% /0.85% /1.02% of active expert weight bytes. Actual DRAM savings can be lower because of caches and unwritten padding. This is not a route to doubling MoE bandwidth by avoiding intermediate traffic.

The larger opportunity is better use of existing bandwidth and shorter stage/tail idle periods. The three archived NCU pairs have56.36–75.54% DRAM peak utilization. Holding their measured bytes constant gives the following arithmetic:

| Layer | Profiled FC1+FC2 µs | Conditional time at90% peak | Conditional saving | Saving at100% peak |
|---|---:|---:|---:|---:|
| 0 | 118.304 | 95.635 | 22.669 µs (19.16%) | 32.233 µs |
| 19 | 98.336 | 75.063 | 23.273 µs (23.67%) | 30.780 µs |
| 39 | 76.800 | 53.554 | 23.246 µs (30.27%) | 28.601 µs |

Formula: `sum(duration_i * observed_peak_fraction_i / assumed_fraction)`. The90% scenario is deliberately ambitious and unproven;100% is an accounting reference, not a service bound. NCU can serialize PDL, these samples use older64-real inputs/source, and the successful oracle set uses63-real+1-padding. Do not subtract this table from the new unprofiled125.389/105.573/86.426-µs native medians or multiply three samples by40 layers.

Historical pure-decode profiling attributes3.957ms to the two GEMMs and0.356ms to activation per replay, about45.8% of its9.408ms summed kernel time together. Historical mixed D58/P6 profiling has7.700ms GEMMs and1.475ms activation, about31.1% of29.501ms summed kernel durations; concurrent spans make these sums differ from wall time. These identify substantial covered work, not available savings. Activation arithmetic, quantization boundaries and finalization remain necessary even when their launches disappear. No current evidence establishes a whole-service gain from this design.

## 5. Smallest implementable gate

1. **One native expert's FC1 first.** Use layer0, expert205, exactly eight archived real routes: token indices0,1,3,23,24,47,57,62. Keep the archived quantized input/scales and native packed expert weights. Start with up128/gate128 and original M64/K32 operations. Export every FC1 E4M3 byte and FP32 scale and compare against `expert-metadata-rebase/oracle-layer0-real.pt`, expanded indices `[2,11,30,186,193,378,457,499]`. No synthetic oracle, re-routing or BF16 replacement. If this fails, preserve maxabs/bitwise differences and localize accumulation versus quantizer; do not proceed to timing or silently accept native-looking dtype.
2. **One four-CTA cluster.** Extend to all four activation blocks and all2048 FC2 outputs for that same expert/token group. Compare FC1 bytes/scales, activation bytes/scales and FC2 BF16 separately. Then layer19 expert4 and layer39 expert229 provide two additional native eight-route samples. Use archived one-route and multi-token-group experts to test inactive lane masks and group boundaries; all nonparticipating buffers/canaries must remain unchanged. Keep each compiled candidate artifact immutable.
3. **Graph ownership and changing inputs.** Reuse the same captured graph with changed native expert counts, empty groups and worst-capacity routing. Retain all descriptors, streams, workspaces, graph outputs and pools. Check native IDs/weights, each route's unique owner, FP8 boundaries and final ordered routed BF16. The15 successful native oracle files cover real/reversed/left-only/right-only/worst-global288 routing cases, but the first single-expert gate cannot substitute for these full-routing checks.
4. **Only then test complete MoE.** Compare original native fullcall, the same new arithmetic split into globally staged phases, and the cluster-fused body, all including identical prepare/routing/finalizer costs and same original PDL control. The split new-body arm distinguishes arithmetic quality from scheduling benefits. Preserve exact intermediates before unprofiled timing. Require at least5% complete local MoE improvement against the original native call, not against either failed partition version. Count shared-memory spills, DRAM bytes and actual cluster occupancy in a separate diagnostic after timing, never as a hidden change to the timed configuration.

No GPU implementation has been authorized in this CPU phase. The concrete next request is a bounded **FC1-only arithmetic equivalence probe**, followed by one-expert cluster validation if it passes, rather than immediately writing a production persistent MoE. The 5% continuation gate does not relax the user's all-concurrency1.5× target. A passing local kernel would still require40-layer native state/RNG checks, full1088 service plus WER/SIM, repetitions and all concurrency points.
