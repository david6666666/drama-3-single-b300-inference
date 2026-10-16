# Independent resident activation/DSM lifetime review

CPU/source/AOT scope only; no new GPU execution, native-chain pass, or performance claim. Reviewed `resident_fc1_pair.py`, `resident_gmem_c.py`, `resident_resources.py`, parent TMEM review, and the explicitly owned `resident_dsm_gather.py`.

## Actual current byte ranges

All offsets are relative to the outer shared byte Array; base alignment is1024, total dynamic allocation78080.

| Object | Half-open byte range | Lifetime |
|---|---:|---|
| Borrowed native stage scratch/barriers | 0:72704 | Each fully drained native stage |
| FC1 up/gate bytes, logical[8,256] | 72704:74752 | Both FC1 stages through all activation readers |
| FC1 original scales, logical[2,8] | 74752:74816 | Same as FC1 bytes |
| Full gathered activation bytes[8,512] | 72704:76800 | After activation convergence through all FC2 stages |
| Full gathered scales[4,8] | 76800:76928 | Same as full activation |
| Local activation bytes[8,128] | 76928:77952 | Activation writes through all peer DSM reads |
| Local activation scales[8] | 77952:77984 | Same as local activation |
| Outer TMEM address slot | 78016:78020 | One outer allocation through final deallocation |

The local activation output is disjoint from both live FC1 inputs and full-gather destination. Thus no activation warp overwrites another warp's unread up/gate/scale. The full-gather region does alias FC1. It must not be written merely because an individual warp has finished activation. The helper's first full512-thread CTA barrier establishes **all** activation readers are finished. Its following ordered cluster arrive/wait publishes local output before any peer loads and converges all four ranks before overwriting old FC1 storage.

## Executable handoff

`resident_dsm_gather(local_act, local_sf, full_act, full_sf, valid_rows, tid)` takes explicit shared Arrays, no hardcoded offsets. All512 threads of all four CTAs call, even zero rows. Four-CTA outer cluster does not change native GEMM cfg.cluster_m=1 or TMEM ownership.

1. Full CTA convergence; ordered aligned cluster arrival/wait. No aliased store precedes it.
2. First256 threads gather16 bytes each: warp=token, eight lanes per peer, so each token obtains four contiguous128-channel fragments. First32 threads copy scale bits in rank/K-block-major, token-minor order. Mapping uses the installed `prims.mapa` distributed-shared pointer primitive; destination is local shared memory. Invalid rows produce zero without reading uninitialized peer padding.
3. Full CTA convergence; second ordered aligned cluster arrival/wait. Every DSM reader has completed before local fragments may be overwritten or any CTA exits. This also publishes every rank's full arrays before FC2.

Caller then uses the separate shared-B/SF adapters, which stage native s128b UMMA operands and preserve native FP32 scale/dequant order. They do not mutate the pinned source. Full arrays must remain live until the last FC2 consumer drains. The helper contains no allocation, TMEM instruction, barrier reinitialization, or floating arithmetic. It does not replace required native producer/MMA/epilogue drains or outer final cleanup.

## Actual budget and evidence

`dsm-gather-aot-result.json` is from an actual sm103a compile of cluster(4,1,1), block512 and the **whole** outer78080-byte Array at1024alignment. Generated host MLIR requests78080 dynamic bytes; cubin reports1024 static bytes,24 registers, zero stack/local. Total is79104 bytes (77.25KiB), leaving2816 bytes below81920. Static and dynamic are both counted. This is a full allocation/layout probe, **not** the compiled six-stage resident chain: only the eventual complete chain's host request plus cubin/static/resource ledger can establish its final80KiB budget and occupancy.

CPU ownership checks cover valid_rows0..8, all4096 destination bytes exactly once, legal peer ranks and offsets, and scale row order. SASS contains two UCGABAR_ARV/WAIT sequences and remote16-byte reads/localSTS.128. Initial CPU type-join error (Uint32 versus Int32 after load) is preserved separately; explicit Uint32 bit representation resolved it, with no GPU attempted.

Remaining gates: actual full chain compile/resource audit; full15-oracle FC1/activation/FC2 native boundaries; zero/partial8-route cases; unchanged sources/global guards and RNG; changed-input captured replay; repeated stage/barrier lifetime and cluster completion. No timing until those gates pass.
