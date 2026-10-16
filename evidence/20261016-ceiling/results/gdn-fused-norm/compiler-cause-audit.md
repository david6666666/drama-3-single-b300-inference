# GDN recurrence + gated norm: compiler audit

Original exact failure remains authoritative (`first-layer-result.json`), with no timing run. The subsequent C2 protocol is a separately declared numerical diagnostic.

The uniquely matched installed native BV32/one-warp cache entry is archived in `A-native-cached.*`; its exact cache path and hashes are in `native-A-cache-candidates.json`. The signature uses BF16 state, K128, V32 tile, mixed row stride8192, a/b stride12352 and num_warps1. B uses the same upstream source at BV128/four warps.

The A and B TTGIR show different Q/K L2 reduction layouts. A loads four entries per thread and reduces within one warp; B loads one entry per thread across four warps. This establishes a compiler reduction-order difference under a tile-only change. It does not individually attribute every changed output element to one operation.

There was also a concrete flaw in the initial C prototype's attempt to preserve native contraction. Native B PTX contracts local `h*k` and `h*q` accumulation into FMA chains, in addition to the rank-one state update. C's global `enable_fp_fusion=False` prevented those reduction contractions; explicitly restoring only the rank-one FMA was insufficient. C2 restores global native fusion and fences normalization square/mean/epsilon with explicit `mul.rn`/`add.rn` inline assembly.

All native BF16 tensor boundaries are visible as truncation followed by extension in C/C2 TTGIR: beta, raw recurrence output, normalized values and weighted normalized values; stored recurrent state and final output are BF16. No expert count, model equation, weight precision or sampling change is involved.

On the fixed first-layer input, C2 state and raw output now exactly match native B. C2 normalization differs from native B at one BF16 element, consistent with the remaining normalization reduction layout difference; the separate per-head numerical/reference gates pass. This does not make C2 bitwise exact against native A. The independent CPU FP64 reference preserves the model's BF16 tensor boundaries and is not an emulation of GPU reduction order.

The initial C's worst per-live-head relative RMS was2.98e-4 for state and2.54e-4 for raw output, failing the new1e-4 diagnostic bound; full-slot or whole-pool RMS averages hid that local maximum. C2's first-layer maxima versus A are3.09e-5(state),6.92e-6(raw),3.41e-5(norm). All nonparticipating slots are exact. Full30-layer and graph checks are still required before timing or promotion.

The expanded run subsequently failed at its first B64 layer on a newly prepared real fixture. `expanded-result.json` is `NUMERIC_FAIL`; later layers, full graphs and all timing were stopped. C2 recurrence state/raw still exactly match native B, but three raw-output heads and two normalized-output heads exceed the predeclared1e-4 A-drift bound; one normalized head also exceeds the independent-reference degradation bound. `expanded-failure-localization.json` reports absolute magnitudes alongside these relative failures. The raw worst case is one differing element of7.45e-9 in a head with RMS4.54e-7. The norm worst case is one differing element of9.54e-7 in a head with RMS2.31e-4; its FP64 reference RRMS increases from0.001766 to0.001835. These observations do not authorize threshold changes. No runtime change or performance promotion was made.
