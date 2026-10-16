# Eight-node MoE counter result

Completed exactly eight serialized NCU node records across L0/real and L39/real, native and metadata-ready. All404 exact/ownership checks passed; each kernel required four replay passes. All150 frozen input hashes still match. GPU2 is released; final process audits are empty. No whole-graph/PC sample, runtime promotion or performance claim was made.

The first launcher stopped after successful L0/native sampling because it expected `.ncu-rep`; NCU2026.3 exported `.ncu-repz`. Its original failure, command and report remain untouched in `../expert-m128-ncu-node`. The v2 launcher accepts exactly one compressed/uncompressed report and ran only the remaining six records; the first two were imported on CPU without remeasurement.

| Layer/FC | Native → candidate µs | DRAM read MB | Total DRAM TB/s | Active CTA/SM | Eligible warps/active cycle | Issue active% | Long-scoreboard ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0/FC1 | 75.744 → 73.056 | 432.35 → 432.25 | 5.836 → 6.047 | 1.00 → 1.95 | 0.30 → 0.20 | 22.71 → 16.41 | 12.54 → 30.78 |
| L0/FC2 | 43.424 → 40.000 | 216.41 → 216.30 | 5.259 → 5.801 | 1.00 → 1.96 | 0.33 → 0.32 | 26.29 → 24.31 | 9.90 → 19.21 |
| L39/FC1 | 48.800 → 46.272 | 241.24 → 246.83 | 5.140 → 5.535 | 1.00 → 1.94 | 0.36 → 0.26 | 26.83 → 20.08 | 9.40 → 24.01 |
| L39/FC2 | 29.248 → 27.008 | 120.97 → 120.78 | 4.485 → 4.844 | 1.00 → 1.94 | 0.39 → 0.41 | 29.84 → 29.12 | 8.14 → 14.17 |

These are actual current-fixture counters. Candidate residency is1.94–1.96 CTAs per active SM cycle versus native1.00; achieved warp occupancy rises from21.84–23.53% to35.59–36.46%. That rules out the earlier unsupported claim that the TS source register hints force a single CTA. It does not establish that more occupancy alone fixes memory supply.

FC1 has fewer eligible warps and lower issue-active percentage despite increased residency on both layers. The long-scoreboard ratio increases from12.54→30.78 (L0) and9.40→24.01 (L39). FC2 eligible counts change little (.33→.32 and.39→.41), with smaller but still elevated long-scoreboard ratios. This supports investigating the dependency of ordinary L1TEX-backed loads and their consumers. It does not identify TMA-stage starvation, a particular load PC, or a wall-time fraction: these ratios are normalized per issue-active cycle, and lower issue activity itself affects their denominator. Aggregate barrier and short-scoreboard ratios are small by comparison, but are not a complete model of asynchronous dependency cost.

DRAM reads stay within−0.154% to+2.318% across arms. Native reads are close to one logical active-expert weight copy on all four records. The candidate L39/FC1 reads2.318% more while running5.180% shorter in node replay; reduced repeated-weight traffic therefore is not the explanation. Aggregate L2 read-sector equivalents change only−1.780% to+2.515%; there is no evidence for a large eliminated L2 transfer budget either. The per-kernel write differences must remain visible, especially L0/FC2; writes are not pure output stores under multi-pass replay/cache eviction. `summary.csv` retains both reads and writes and the exact denominator values.

Candidate isolated node reductions are3.549%/7.885% at L0 and5.180%/7.659% at L39 for FC1/FC2. Their sum is not a full-call or PDL gain. The separately completed unprofiled one-screen full-MoE reductions were only2.729% and4.145% and failed the5% continuation gate. No three-round or service result is inferred.

The conditional memory budget can now be recomputed from current bytes and duration: `duration × (1 − measured_DRAM_peak_percent/90)` is the hypothetical saving at90% of that record’s implied peak, assuming fixed bytes and no other limit. This remains a scenario, not recoverable critical-path time. `summary.json` records it for each observation; neither these scenarios nor serialized kernel deltas may be added to the complete-call gain. The next substantial hypothesis still needs12.075µs/L0 and7.994µs/L39 to reach10% versus native; current node observations do not establish that budget as achievable.

**Recommended next discriminator: at most two PC-sampling records, L0/real FC1 native versus candidate, only after source review and authorization.** The repeated FC1 pattern is the clearest unresolved issue: more resident work yet fewer eligible warps, with the same weight volume. Attribute long-scoreboard samples to exact SASS PCs and native/TS load/consumer instructions before proposing a stage/metadata/scale-load change. Use a fixed same-case graph and preserve all predecessor/PDL/exact gates. If sample count is inadequate for these~73–76µs kernels, report inconclusive rather than silently repeating or changing sampling periods. No broad SourceCounters set or tile sweep is proposed.

Whole-graph profiling would answer a different question—whether actual PDL interaction/launch overlap explains why the complete-call gain is smaller—but does not identify which ordinary load causes the FC1 scoreboard dependency. It is deferred while the source/PC question is addressed. A later graph result still needs the same complete metadata→FC1→activation→FC2→finalizer boundary and typed dependency proof. No tail duration is inferred from static waves or average issue/stall counters.

This is diagnostic evidence for the next native structural change. The user’s all-concurrency throughput and WER/SIM goals are unchanged. Original negative correctness/performance gates remain preserved.
