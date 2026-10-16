# Native ordered finalization plus residual RMSNorm

This artifact retains all eight expert branches and the original FP32 ordered weighted accumulation, BF16 routed cast, existing BF16 shared-expert FMA, BF16 residual boundary and RMSNorm expression. It does not modify the serving runtime. Source is frozen95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b; both arms ran on GPU4, UUID GPU-4ae7e013-1aac-31d8-8b8e-c27530f1c6ed.

The final driver passes1355 checks:15 original L0/19/39 FC2 fixtures, archived/reversed physical maps, a missing branch and all branches missing; eager and three replays of the same graphs with changing inputs/maps; original native finalized values, fused output and residual, read-only tensors, canaries and RNG. Counts include provenance/ownership assertions, not1355 separate model runs. Shared/residual/gate values are deterministic fixture-derived inputs, not captured live shared-expert outputs. This is an operator-only contract, not full-model or audio-quality qualification.

| Real case | Native finalizer + existing fused norm + restore µs | New fused finalizer/norm + restore µs | Saved µs |
|---|---:|---:|---:|
| L0 | 8.240000 | 7.182720 | 1.057280 |
| L19 | 8.218880 | 7.182400 | 1.036480 |
| L39 | 8.306240 | 6.505280 | 1.800960 |

This is one predeclared AB screen of100 retained graph replays per arm/case, after five warmups. Both graphs contain the same residual copy to keep repeated work fixed. Every real case misses the2.5µs absolute saving gate, so the result is **FEASIBILITY_FAIL**. No three-round, full-MoE, full-model or serving timing follows. Multiplying these local savings by40 layers would be only an illustrative0.041–0.072ms, not a demonstrated decode gain; it cannot bridge the C64 target gap.

The fused kernel AOT uses48 registers and1024 shared bytes, no stack/local spill. PTX retains an entry `griddepcontrol.wait`, BF16-input FP32 `fma.rn.f32.bf16`, the subsequent BF16 cast and `fma.rn.bf16` shared combination. The graph snapshots preserve actual nodes/edges. There is no live FC2 producer here, so neither launch_pdl nor WAIT proves overlap with FC2. Map values are trusted fixture data (-1 or a legal physical row), not a general runtime validation API; live workspace lifetime and alias safety remain separate integration obligations.

## Preserved setup and reference failures

- `../native-finalize-norm/cpu-aot-first-failure`: zero-result inline assembly triggered a Triton lowering assertion. The installed `tl.extra.cuda.gdc_wait` builtin supplies the expected dummy result; original source and failure remain. A later CPU-only result serialization failure is separately retained.
- `../native-finalize-norm/gpu0-preflight-stop`: a foreign process occupied GPU0; no task child or GPU work launched. It was untouched. Both test arms moved to idle GPU4 before execution.
- `../native-finalize-norm/result.json`: Torch reports its UUID without the `GPU-` prefix, so a spelling-only identity assertion stopped before numerical kernels. V2 normalizes only that prefix.
- `../native-finalize-norm-v2/result.json`: the original native-control call incorrectly set `use_routing_scales_on_input=True`, bypassing weighting of unweighted FC2 input. Its first archived native-output gate failed on131069/131072 elements, max difference13.61145, before candidate comparison or timing. Installed launcher C++ line3051 and the original runner establish the correct final boolean=False. V3 changes that argument only; this was a test-harness reference failure, not evidence of fused-kernel numerical degradation.

All25 frozen files still hash-match. The process exited successfully and GPU4 post audit is empty; ownership was sampled every2s, not continuously proven. Original failures and gates are unchanged. Current C64 target6765.407 frames/s and the all-concurrency/WER/SIM acceptance remain unmet.
