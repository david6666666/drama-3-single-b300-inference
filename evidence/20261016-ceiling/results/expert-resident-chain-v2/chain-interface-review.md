# Full resident chain interface review

CPU-only review of the hashes recorded in `chain-interface-review.json`. No blocking mismatch found in the current activation/DSM/shared-B wiring. No full-chain pass or timing claim.

- FC1 up/gate original FP8 bytes and scales stay in disjoint pinned storage until every activation warp completes; full-gather alias is written only after full CTA and ordered cluster barrier.
- Activation Tensor strides match validated helper: [8,256] (256,1), [2,8] (8,1), local output [8,128] (128,1), [8] (1).
- Four local scale ranks gather to [K128 block, token]; FC2 stage indexes stage_info.loop_offset with same K128 block order. Weight scale index retains expert/output-tile/Kblock.
- B uses one complete LoadB warp, 32 AsyncThread producer arrivals, native single UMMA consumer; all lanes stage two16B chunks and async-proxy fence before commit.
- SF uses one complete LoadSfAb warp, 32 AsyncThread arrivals, four native epilogue warps (128 consumers); all32 scale stage elements initialized, one lane writes weight scale unchanged.
- For cfg.cluster_m=1, installed Task CtaLeader signaling tests actual_rank % cta_v_size==0, so all four outer ranks signal their local native pipelines; no accidental rank0-only barrier.
- Rows beyond valid_rows are not read from uninitialized local activation; full gathered invalid rows zero, FC2 B invalid rows zero and SF unused32 slots zero. Global boundary outputs preserve invalid rows through masks.
- Native body may skip on uniform ncta0, but outer activation/DSM barriers and single TMEM allocation/deallocation remain outside that conditional.
- Outer calls one alloc/relinquish before stages, retains TMEM through two FC1/four FC2 stages, then one final dealloc; helper and adapters add no TMEM calls.
- Full/global activation debug exports are not used as FC2 operands. Shared arrays remain pinned across all4FC2 phases; stage scratch bounds exclude them.

Remaining scope and gates:

- Current fixed grid4 and local WorkTile(x+phase4,y0) support one selected expert token tile of0..8 rows, not full256expert metadata. chain_common.prepare now asserts this bound.
- Complete chain actual resource audit and native oracle exact gate are still pending; componentAOT and standalone activation pass do not establish whole-chain pass.
- Actual asynchronous MMA/task drain, stage barrier reinitialization, dynamic register transitions and final TMA output completion need completechain execution; source review only confirms intended placement.

Follow-up: `chain_common.prepare` enforces the bounded input shapes/dtypes and integer valid_rows0..8. The full chain actual AOT now reports CPU_AOT_PASS; GPU exactness remains pending.
