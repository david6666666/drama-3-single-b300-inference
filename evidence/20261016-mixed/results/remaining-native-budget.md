# Remaining native-computation budget

**There is presently no measured, untried native restructuring with evidence sufficient to bridge the remaining gap. This is a prioritization verdict, not a proof that the target is impossible.** Keep full top-8 experts, original FP8 scales, BF16 dense computation and the confirmed WER +1 percentage point / SIM -2 limits.

The latest private-workspace asynchronous-codec candidate is **4360.9165 frames/s**, source `7618212e15e7506bb20d57372d6700fe837988a5`, one complete C64/1088 run, **quality pending at this analysis**. Reaching 6765.4073 requires another **55.14% throughput gain**, equivalent to **35.54% less elapsed time** for equal work. See [candidate result](codec-private-gpu0/benchmark/summary.json) and [launch](codec-private-gpu0/launch.json). The earlier 4083 frames/s point needed +65.70% throughput / -39.65% time. Neither feasibility point is all-concurrency acceptance.

## Existing pure B64 trace

Three captured decode graphs contain 1298 kernels each. Mean summed kernel duration is **9.408 ms**, while elapsed graph spans are **8.728/8.791/8.749 ms** because kernels overlap. The following disjoint categories partition the **sum**, not critical-path time. They come from the older native projection trace; input quantization/casting has subsequently been fused, so that row is not an untouched opportunity.

| Category | Mean summed ms/graph | Share of summed kernel time |
|---|---:|---:|
| Native expert FC1 + FC2 | 3.956 | 42.05% |
| FI expert routing + activation + finalize | 0.960 | 10.21% |
| All BF16 dense projections, including split-K reduction | 1.898 | 20.18% |
| GDN recurrence + convolution + gated norm + z contiguous copy | 1.033 | 10.98% |
| Attention + reduction + preparation + backbone gate reshape copy | 0.398 | 4.23% |
| Residual/RMS norms | 0.339 | 3.60% |
| Sampling kernels | 0.077 | 0.82% |
| Old MoE input quantization/casts, already fused subsequently | 0.257 | 2.74% |
| Other embedding, pointwise, shared combine and request work | 0.489 | 5.19% |

The 1.898 ms dense category is spread across 235 GEMMs: backbone output projections 0.424 ms; GDN input 0.350 ms; packed shared/router input 0.208 ms; shared-expert output 0.132 ms; backbone QKV 0.091 ms; tower FFN gate/up + down 0.304 ms; tower attention projections 0.250 ms; tower embedding/hidden projection 0.075 ms; semantic/code heads 0.064 ms. These counts follow the existing model's projection order; reduction kernels are assigned to their preceding GEMM. No single unoptimized repeated projection dominates.

Uncovered intervals between all GPU kernels total only **46.8-48.1 us/graph (about 0.54%)**. Counting launches and multiplying by a guessed host launch cost would be incorrect. The full decode tower after semantic sampling spans 1.323-1.326 ms, about 15% of a decode graph; nine residual-code steps have real sequential dependencies. The admission-only joint tower proposal can save at most about 1% of the old measured service interval even if its entire old-decode tower cost vanished: [deferred design](artifactproposal.md).

Trace: [raw](../drama3-scale150-20261015/native-bf16-projections-b64/trace.json.gz), [operator table](../drama3-scale150-20261015/native-bf16-projections-b64/table.txt), [timing](../drama3-scale150-20261015/native-bf16-projections-b64/timing.json). Raw trace SHA256: `e0b7af65c92c32ed8548f3f167478610294540f797d3be3ff97f315c60db2ecc`.

## Service budget: illustrative historical accounting only

The earlier source `a25c68d` 384-request diagnostic has a 9.9984 s interval, pure decode event spans 6.0598 s (60.61%), mixed backbone including teacher tower 1.9687 s (19.69%), and codec 1.4542 s (14.54%). [Phase analysis](mixed-timeline-phase-analysis.json), [experiment](mixed-timeline-experiment.json).

At fixed remaining work, eliminating that codec cost entirely gives an ideal **1.170x** speedup; eliminating all mixed work gives **1.245x**. “Codec +17%” is a throughput ceiling in this illustrative model, not a 17% time share. Even deleting both costs entirely gives **1.521x**, roughly 6208 frames/s when applied to the old 4083 point. In contrast, halving both gives only 1.207x. These impossible zero-cost scenarios expose the size of the gap; they are not implementation predictions.

Using the same historical proportions as an optimistic accounting exercise, doubling every BF16 dense projection in pure decode gives only about **1.065x service**; doubling the entire GDN group gives **1.034x**; doubling all attention/preparation gives **1.013x**. Even halving every non-FI-MoE category in pure decode would give only about **1.17x** in that historical service mix. Kernel overlap and changed scheduling prevent treating these estimates as strict Amdahl bounds. **Do not reuse this old synchronous profile as a lower bound or projected stage breakdown for the latest 4360.9165 asynchronous candidate.**

## Priorities and remaining hypotheses

1. **Finish the private-workspace codec candidate's quality evaluation, then measure the surviving service gaps.** This addresses an observed whole-service cost. Its eventual new profile, including startup, continuous admission and drain, should determine whether CPU waits, codec delivery or sparse useful GPU work still dominate. A Rust rewrite has no present evidence for a 55% gain.
2. **Native decode fusion has a bounded local opportunity.** Conv/preparation, strided z input and gated norm could remove copies/materializations, but cannot remove the recurrent BF16 state update. The old z copy is only 0.111 ms and conv plus gated norm 0.216 ms. GDN value-tile sweeps (8/16/32/64, multiple warp counts) and the FI backend have already been tested; repeat flags/tiles are not a new hypothesis. Dense projection tiling also tested 144 real-input configurations without beating existing dispatch. A genuinely new fused epilogue must first demonstrate a controlled layer-level gain; the category budget cannot substantiate a target-sized service gain on its own.
3. **A larger native MoE improvement would matter, but the easy memory-reuse explanation is contradicted by counters.** B64 FC1/FC2 read about one active expert-weight copy, with low tensor activity and substantial bandwidth. Real mixed M1344 changes to a larger tile and still reads only **1.011-1.017x** unique active-expert FP8 weight bytes, including activation/scale reads. Actual FI routes were captured with exact-output checks. See [B64 counters](ncu/README.md), [mixed counters](ncu/mixed/README.md) and [mixed machine-readable evidence](ncu/mixed/summary.json). This does not prove peak performance, but it rules out asserting a large gain from deleting repeated HBM expert reads. BlockMajorK/shuffling is already implemented and previously tested; no further arbitrary layout/autotune sweep is justified by these data.

The backbone already shares QKV/GDN projections across mixed rows and packs shared gate/up, router and shared gate into one input projection. Gate/up tensor copies retained in model storage are not duplicate forward GEMMs. Cross-layer batching cannot freely process multiple future native frames because each frame depends on previous sampled semantic/residual codes and recurrent state. Any larger speculative/temporal scheme would need new correctness and acceptance evidence; no measured acceptance rate or speedup currently supports it.

**Conclusion:** continue the evidence-backed service/codec work and tightly scoped native fusion where its measured budget warrants it. There is no supported single remaining 20%-plus service optimization in the enumerated non-MoE categories, and no current evidence that their combination reaches the target. The hardware-counter samples and historical profiles do not establish an impossibility result or all-concurrency lower bound.

Follow-up: the complete private-capture quality group now passes: WER 1.590890%, SIM 64.191350, all 1088 outputs scored. The all-concurrency throughput/quality campaign remains incomplete.
