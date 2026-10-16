# Independent native FI reference and padding-domain diagnosis

This is a separate diagnosis of the retained failure in `fi-mixed-validation.json`; that original all-sequence 0.01 gate remains FAIL. Frozen source `1618ad4b2c7591465a47ca0dad8100e782820cd2`, same D64/P1280/span256 real 58:6 inputs, compaction off. No timing or service claim.

All 30 identical-input real-sequence GDN output/state comparisons stayed below the separately declared 0.01 diagnostic threshold: maximum output RRMS 0.007224 and state 0.005890. All 503 recorded diagnostic/ownership/replay/intervention checks completed. FI eager and captured graph outputs/all mutable state were exact. This does not establish final model accuracy: complete-trajectory decode hidden was exact versus FLA, while returned prefill hidden RRMS was 0.091184. Full WER/SIM remains required.

Layer0 independent FP32 token recurrence starts from the same stored BF16 initial state, uses the exact caller BF16 Q/K normalization boundary promoted to FP32, original BF16 V promoted to FP32, original FP32 beta and the same vectorized FP32 `exp(decay)` consumed by FI. FLA instead evaluates cumulative log-gates in chunks, so finite-precision operation ordering is expected to differ. The reference does not emulate either chunked algorithm.

| Sequence | Tokens | FI output vs FP32 RRMS | FLA output vs FP32 RRMS | FI state vs FP32 | FLA state vs FP32 |
|---|---:|---:|---:|---:|---:|
| real 0 | 184 | 0.002322 | 0.002422 | 0.003032 | 0.002977 |
| real 1 | 190 | 0.002340 | 0.002419 | 0.003008 | 0.003323 |
| real 2 | 152 | 0.002329 | 0.002275 | 0.002654 | 0.002831 |
| real 3 | 243 | 0.002215 | 0.002275 | 0.002714 | 0.003102 |
| real 4 | 175 | 0.002328 | 0.002075 | 0.002489 | 0.002748 |
| real 5 | 177 | 0.002315 | 0.002100 | 0.002850 | 0.003077 |
| padding 8 | 159 | 0.022450 | 0.014388 | 0.004034 | 0.003313 |

The failing private-padding sequence has reference output RMS 0.044487 versus about 0.27 on real prompts. FI error RMS is 0.000999 and maximum absolute error 0.146903; FLA error RMS is 0.000640 and maximum absolute error 0.107003. FI is worse for this artificial sequence, so the discrepancy is not dismissed as only a small denominator. Both maximum errors occur at padding token63/head7/channel79: FP32 reference0.939034, FI1.085938, FLA0.832031. All Q/K/V/decay/beta values become constant after the first three convolution warmup rows. The first 64-token chunk dominates the discrepancy (FI reference RRMS0.034598, FLA0.022581); later chunks drop to FI0.007968/0.005516 and FLA0.003402/0.002480. This pattern is consistent with differing chunked arithmetic on repeated padding, but does not by itself prove the precise rounding source.

Padding isolation follows the implemented ownership boundaries:

- `source-compact-1618ad4/src/drama3/mixed_graph.py:133` constructs distinct decode/prefill/private-padding slots and rejects overlap; arriving state reset addresses only layout sequence slots.
- `source-compact-1618ad4/src/drama3/packed_prefill.py:85` masks convolution reads to each sequence start/end and writes its own indexed state.
- `source-compact-1618ad4/src/drama3/mixed_model.py:78` passes the same packed boundaries as query and KV cumulative lengths into causal flash attention. Decode attention uses its own slot batch. Teacher-forced tower attention receives the same PackedLayout.
- `source-compact-1618ad4/src/drama3/gdn.py:88` passes cu_seqlens plus indexed state slots to native FI. All nonparticipating slots were checked exactly.
- `source-compact-1618ad4/src/drama3/moe_input.py:11` quantizes each token with independent 128-channel scale groups; router logits are per token. `model.py:537` calls FI with 256 local experts and full top8. Its interface defines routing as per-token softmax/topk/renormalization, with no shared batch activation scale. Dense projections, residual normalization and shared-expert combination operate on each row.

A direct causal intervention strengthens the code analysis: zeroing only the trailing private-padding GDN outputs after every one of 30 layers left all real decode/prefill final hidden vectors and every primary real conv/GDN/KV slot bit-exact to the ordinary FI trajectory. Padding state itself was allowed to differ and never admitted into a real slot. This was one fixed real shape/trajectory, not a universal correctness or quality proof. No intervention was added to the framework.

Retained evidence: `fi-mixed-reference-experiment.json`, `fi-mixed-reference-command.json`, `fi-mixed-reference.json`, `.log`, `diagnose_fi_mixed_reference.py`, `fi-mixed-layer0-reference-inputs.pt`, and CPU-only `fi-mixed-padding-error-shape.json`. Original failed report hash is captured in the separate command metadata. GPU2 returned to4MiB/0% after PID239838 exited.
