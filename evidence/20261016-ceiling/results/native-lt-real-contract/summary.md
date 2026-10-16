# Restricted Lt real-input investigation: no replacement selected

Both fixed algorithm descriptors pass all84 eager full-output byte comparisons
against native F.linear, and84 captured-graph output/input/weight/guard/RNG
checks. These are 30 real calls of [64,12352,2048] non-split and54 calls of
[64,2048,4096] FP32-workspace split-K. Real strides and call order are preserved.
No new dtype or weight conversion was used; the capability flag discrepancy
is unresolved and this artifact is not qualified for native runtime promotion.

One100-replay feasibility per family:

| Family / calls | F.linear sequence ms | Explicit Lt sequence ms |
|---|---:|---:|
| Non-split /30 |0.331939507|0.331991692|
| FP32 split-K /54 |0.412404785|0.412598724|

Both are essentially the same speed in this single screening and neither was
selected. The later whole223-call sequence runs F.linear on both arms because
replaced_calls=0. Its1.550477→1.523957ms difference is **A/A variation**, not an
optimization or a0.0265ms saving. The recorded0.3ms continuation gate is not met;
there are no repeated trials, full-model changes or service/quality claims.

The223-call sequence covers intercepted drama3.model.linear calls. Direct
matrix operations inside table_projection and other modules are not included;
this is not all GPU dense work. At the opening budget measurement it costs
1.550888ms; the two selected shapes together cost0.742328ms. Both are diagnostic
budgets on this input/position, not percentages of a current service profile.

The original zero-candidate strict-filter run, eight range probes, initial C++
context failure and corrected library identity all remain preserved. The native
FP16/BF16 capability semantics must not be silently reinterpreted. No further
SASS qualification is needed for a candidate with no measured benefit.

GPU0 task exited and released memory. Process ownership was sampled every2s,
which is not continuous exclusivity proof. The C64 target remains6765.407fps;
best complete-quality observation remains4726.771fps.
