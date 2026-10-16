# Final independent result after the reference correction

The corrected v3 passed all1,355 native exact/lifecycle checks. Its single AB100 operator screen still failed the predeclared2.5µs savings gate on every real layer: L0 saves1.057µs, L19 saves1.036µs, L39 saves1.801µs. I independently recomputed each subtraction and gate from result.json. These are separate outcomes: the original v2 failed its erroneous native reference control; the corrected v3 passed exact comparison but failed the performance budget. No full-model timing, three-round claim or runtime promotion follows.

# Review correction: native reference Boolean ABI

The initial CPU review missed a native-reference ABI error. The original harness passed `(enable_pdl=True, use_routing_scales_on_input=True)` despite holding unweighted FC2 outputs. Root's first `native_archive` control caught the mismatch before candidate comparison or timing. This is a harness control failure, not evidence of fusion numerical degradation. The original review and experiment failures are retained separately.

The installed C++ signature at `trtllm_fused_moe_kernel_launcher.cu:3051–3055` names the last Boolean `use_routing_scales_on_input`; lines2714–2715 set expertWeightsPtr to null when it is true. The original block-scale runner at `runner.py:2384–2395` passes False. Independent AST inspection confirms v2 used True/True and v3 now uses True/False, while fused_finalize.py is byte-identical between v2 and v3. That correction is source-supported and must pass the unchanged native-archive gate before any candidate or timing result is accepted.

The following earlier source/lifecycle observations remain useful, but their no-blocker sentence is superseded by this correction. No GPU work was performed by this review.

---

# Independent CPU review: native finalizer + residual RMSNorm

No blocking issue was found for the explicitly bounded artifact-only exact gate. This is not a runtime or numerical PASS. The final protocol/source/CUBIN hashes and15 actual fixture domain checks are retained in the companion JSON; CUDA was not initialized by this review.

The candidate visits original top8 order, converts each BF16 probability/value into FP32 semantics, uses explicit FMA, rounds the routed result to BF16, then follows the existing BF16 shared-gate FMA, residual BF16 round, and FP32 RMSNorm sequence. CPU-compiled PTX actually folds the inputs into `fma.rn.f32.bf16`, then performs an independent BF16 conversion; the shared combination remains `fma.rn.bf16`. Native source at `trtllm_fused_moe_dev_kernel.cu:678–716` skips only map−1 and accumulates original order. Hardware comparison to the native finalizer remains authoritative; copied source expressions alone do not prove identical compiler reduction layout.

Independent CPU checks verified all15 archived expanded maps contain512 unique values within0..2303, reversed physical maps remain in bounds, and native logical FC2/routing-weight fixtures are finite BF16. Thus current full-map, reversed-map, single−1 and all−1 cases are safe. The wrapper is not a general map-domain or alias validator: other negative indices or out-of-range positive values are outside the declared contract. It should not be promoted as a general runtime-safe API on the basis of these fixtures.

Fixed tensor addresses and retained owners support reused graphs across all map/input variants. Both timing arms include the same residual restore; the measured scope is finalization+existing shared/residual RMSNorm, not the entire MoE or a service loop. Post-timing output/residual/native-archive, immutable inputs, guards and RNG checks were added during review, closing the original harness gap.

The candidate has an emitted entry `griddepcontrol.wait` and launch_pdl=True. It has no early explicit dependency release; CTA completion can release dependents, and this terminal operation does not require early release for correctness. Actual graph nodes/edges are saved. The current producer is a residual-copy operation with ready frozen FC2 tensors, so this gate cannot demonstrate live FC2→fused-op PDL overlap or validate future runtime workspace aliasing. No such claim is made.

Deterministic shared/gate/residual values are derived from native input/router, not actual captured shared-expert outputs. The15-case exact gate and the single predeclared2.5µs complete-scope screen remain required. A passing operator result would still need actual model, concurrency and WER/SIM validation before service promotion.
