# Native MoE residual fusion: exact gate and isolated timing

Frozen source: `6ff8d8067fb6e96804f49fd4f0c9020fd315e1fc`. GPU6 UUID `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`. Native fulltop8, unchanged expert scales, BF16 dense/state. Default flag remains off.

Result: **PASS, 2374 exact checks.** All 40 destination norms with 10 static inputs each; all40 live MoE boundaries in each full graph; graph/eager output and full conv/GDN/KV/request/RNG state equality; untouched slot ownership. Includes existing shared-expert stream compatibility.

The original `da91321` FAIL remains in `moe-residual-validation.json`; the causal BF16 FMA correction is documented in `moe-residual-fma-fix.md`. The original validation driver hash is unchanged.

| Case | Control median ms | Fused median ms | Reduction ms | Reduction |
|---|---:|---:|---:|---:|
| b64 | 7.192544 | 7.151616 | 0.040928 | 0.569% |
| d58-p6-d64p1280 | 28.315647 | 28.022783 | 0.292864 | 1.034% |
| b64-shared-stream | 7.124992 | 7.059488 | 0.065504 | 0.919% |

Five alternating CUDA-event samples after one warmup each, with identical full-state restoration outside events. This is one local feasibility run, not a service speedup claim. The B64 shared-stream row is a compatibility check, not evidence for enabling overlap in the mixed service (mixed still rejects shared overlap).

The mixed reduction exceeds the predeclared 0.1 ms plausibility cutoff. Applied illustratively to the older 1622-decode/197-mixed rich ledger counts, the non-overlap deltas total about 0.124 s (~0.53% of 23.23 s), before service interactions. It cannot bridge the 6765 fps target. No full service run was performed by this agent.

Artifacts: `moe-residual-fixed-command.json`, `moe-residual-fixed-validation.json`, `moe-residual-fixed-validation.log`; original experiment and all failure/diagnostic artifacts retained.
