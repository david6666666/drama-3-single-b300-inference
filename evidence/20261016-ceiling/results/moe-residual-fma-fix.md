# Native MoE fusion: first failure and causal correction

Frozen source `da9132185ed2c503a1e392754e26ff8916d163fa` failed the first static norm0/B1 exactness check. `moe-residual-validation.json` and its log are retained; no full graph or timing ran. The immutable original validation driver is unchanged.

A separate, tiny GPU6 diagnostic loaded only checkpoint layer1 input RMS weight. It reused the declared routed/shared/gate generator; because the original unseeded residual was not saved, it used and saved a new fixed-seed residual. This distinction is recorded in `moe-residual-diagnostic.json`.

The instrumented kernel output matched the original fused kernel exactly. The discrepancy was already present in combined MoE (B1: 542 of 2048 elements; RRMS 0.002085), propagated to residual (20 elements), while running the existing norm on the candidate residual was exact. Thus reduction ordering was not the cause in this probe.

Current baseline `combine_experts_kernel` is compiled with fusion enabled. Its emitted PTX uses `fma.rn.bf16(shared, sigmoid_gate_bf16, routed)`, contracting across the apparent source multiply/cast/add boundary. The initial fusion used `enable_fp_fusion=False` to preserve RMS arithmetic and emitted separate BF16 multiply and add. Source-level casts alone therefore did not preserve the actual baseline result.

The isolated correction explicitly emits that same native BF16 FMA and keeps global arithmetic contraction disabled for residual/RMSNorm. On the saved B1 and B64 tensors, both normalized output and mutated residual are now exactly equal, with zero differing elements (`moe-residual-fma-diagnostic.json`). This is only a causal diagnostic; it does not replace the original failed gate or full-model validation. Original/cache PTX and IR copies are retained under `moe-residual-compiler-evidence/`.

Next: freeze the one-file correction and run the unchanged complete validator to a new result path, retaining the original FAIL. No timing until all declared exact checks pass for each case.
