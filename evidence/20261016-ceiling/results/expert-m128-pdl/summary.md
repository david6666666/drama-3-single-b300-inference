# M128 with PDL: exact, feasibility threshold not met

All 1,498 exact gate checks passed, covering the 15 native oracle cases, live-map ownership, eager/reused graphs, every native intermediate and finalized value, guards/padding/tails, unchanged input/weights/RNG and CPU/GPU CUBIN identity. The fixed nonzero-grid FC1→FC2 empty `ncta=padded=0` gate passed both eager and captured replay, then restored the same graph to the exact positive native values.

The complete four-kernel graph contained three actual programmatic dependency edges; the two-GEMM empty graph contained one. These were queried without graph changes. Per-node attribute ID6 queries returned `CUDA_ERROR_INVALID_VALUE`, explicitly retained in the result; typed programmatic edges establish captured PDL behavior. CPU AOT independently confirms one WAIT and two launch-dependents instructions per GEMM, lowered to one ACQBULK and two PREEXIT instructions, and host launch attribute ID6=1.

The single AB feasibility round (five warmups, 100 complete graph calls per arm) did **not** meet the requirement that each real case be at least5% faster than native:

| Real layer | Native µs | M128 PDL µs | Reduction |
|---|---:|---:|---:|
| 0 | 120.781 | 118.380 | 1.988% |
| 19 | 100.754 | 96.892 | 3.832% |
| 39 | 81.966 | 76.790 | 6.315% |

Stress performance is **NOT_RUN**: the 12 stress cases received exact checks only, not timing in this phase. The raw feasibility field `continuation_gate.stress_none_worse5pct=true` is an implementation placeholder used to skip that condition during the three-real-case screen, not evidence that stress latency passed. Raw output is preserved; any later harness must use `null`/`not_evaluated` for this unmeasured condition. The separately executed earlier PDL=False three-round results are not a matched causal comparison with this one-round run.

Therefore the coordinator stopped with `STOPPED_FEASIBILITY_FAIL`. No three-round timing, stress performance claim, production promotion or whole-model/service experiment followed. These single-round results are a screening observation, not a formal A/B gain claim. The post-process audit shows GPU2 empty after both stages.

## Isolated change and exact boundary

Relative to the prior validated M128 candidate, only both GEMMs' `use_pdl`/`do_pdl_wait_for_num_non_exiting_ctas` and the original activation/finalizer enable flags changed from false to true. M128/N8/K32, tile128x8x128, four stages, original top8/FP8 checkpoint bytes/scales, K128 ordered scaling, repaired native epilogue layout and all buffer contracts stayed fixed. The baseline remains the original vendor `(8,5)` prepared body with PDL enabled. No extra PDL=False arm was timed.

Each timed call includes live original routing metadata, FC1, native activation/FP8 boundary, FC2, and the original ordered-top8 finalizer. Common input quantization and logits-to-top8 are outside the fixture. The three real physical B64 samples contain63 live rows and one original padding row, not whole-service traffic. The source is95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b; B300 GPU2 UUID isGPU-bc643dc2-a0d1-5ed6-5ecb-0605ca9f340a.

This global M128 implementation uses CLC persistent scheduling. Whole-zero validity means all payloads skip while work-queue tails/drains remain; it is not a static outer early-return guarantee. Empty activation/finalizer semantics were not assumed or invoked. Positive native activation/scales were restored before replaying the two-GEMM body.

## Reproduction and preserved failures

`protocol.json`, `frozen-inputs.json` (102 frozen files), `pdl_driver.py`, `pdl_gate.py`, `launch.py`, `prepare_protocol.py`, `cpu-pdl-audit.json`, `prepare.json`, `gate-result.json`, `feasibility-result.json`, `launch-result.json`, per-stage logs and ownership audits form the reproducible record. The first CPU host-attribute parser used FC1 SSA IDs for FC2 and failed before freezing or GPU launch; its complete source/log are retained in `cpu-host-parser-initial`. The parser was corrected to the actual independently inspected FC2 SSA IDs; kernel/CUBIN/math were unchanged.

Actual shared resources remain74,616 bytes FC1 and74,664 bytes FC2, zero stack/local/spill. CUBIN hashes: FC1 `7c9bea320d547429ee935f7f43834afd931c0457dd2853cd277af1559984c4aa`; FC2 `a718156952eb2af9cfdbdb020927e95d789f812711e6ff649fe1dc5782ce4419`. PTX register hints do not prove occupancy or runtime register redistribution.

The narrower follow-up is a CPU dependency audit: vendor FC2 has task-local ACQBULK waits and can issue the weight TMA before them, whereas current TS FC2 waits at kernel entry. Removing that entry wait is **not authorized by this result** without proving metadata is ready through the actual graph's dependency chain and retaining activation/SF consumer waits. Current results/source remain frozen.
