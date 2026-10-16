# Native MoE scheduling audit after the resident negative result

The completed resident implementation is exact but slower by 14.16%, 21.51% and 37.11% on the three real layers. The globally staged M128 implementation is exact and only 1.30%, 0.57% and 2.87% faster in its separate experiment. Neither passes the predeclared continuation gate. The measurements do not support another canary, tile or register-cap sweep. Original results remain in `expert-resident-timing-v2/summary.md` and `expert-m128-body-timing/result.json`.

There is not yet evidence for a specific >=10% complete-MoE improvement. The next bounded experiment is to enable the already implemented PDL interface in the globally staged M128 chain, keeping four stages and all arithmetic fixed. This addresses an identified missing scheduling mechanism with a small implementation and exact ownership gate. It does not justify a >=10% forecast. Root authorized this one-variable feasibility in `expert-m128-pdl`; eight-stage tuning is deferred. Prefix-DAG work remains an independent structural direction, rather than assuming a local MoE experiment must succeed.

## What the actual paths establish

The vendor kernel in the archived NCU sample uses `t128x8x128u2_s8_et64x8_m64x8x32...rgTma...dynB`. Its registered tactic remains `(8,5)` in the current reference. The installed mapper's source configuration has eight operand/scale stages, M64/N8/K32, tile128x8x128, two M64 epilogue groups, original native TMA routing and separate activation/finalizer. The M128 candidate keeps the same output tile, K32 MMA and ordered K128 scales, changes only the per-instruction M/epilogue to128, and reduces operand/scale stages to four. All original FP8/FP32 intermediate boundaries and BF16 FC2/finalizer boundaries have explicit exact gates.

Source anchors under installed `flashinfer/prims_ts`:

- `moe/config_mapper.py:1060`: PDL overrides set both `use_pdl` and the dynamic-count wait flag; current map specs are recorded in `expert-m128-fc1-expanded/prepare.json` and `expert-m128-fc2-expanded/prepare.json`.
- `batched_gemm/batched_gemm_config.py:2028`: M64 DeepSeek two-epilogue path uses eight epilogue warps. M128 uses four; actual compiled CTA sizes are512 versus384. This is a real role-count difference, not a measured occupancy improvement.
- `batched_gemm/gmem_ab_resources.py:235–278`: metadata resolves one expert and local row count per token tile; native K128 coordinates and weight expert index remain unchanged.
- `batched_gemm/batched_gemm_tasks.py:140–303`: original load tasks acquire a stage, issue TMA loads, commit, then advance. PDL release is after their work loop, not a change to the GEMM accumulation.
- `batched_gemm/batched_gemm_kernel.py:1790`: the **global M128 path is CLC persistent**. Inactive token tiles are skipped by the work queue, with tail/drain retained. The resident prototype used a separate static mapping; its lifecycle cannot be copied into the PDL explanation.

The four-stage candidate has actual shared allocations74,616/74,664 bytes for FC1/FC2, with zero stack/local/spill in its CUBIN. FC1 has16 K128 iterations per work tile; FC2 has4. Eight stages could allow more FC1 prefetch and, for persistent FC2, potentially cross-work-tile lookahead. It also adds approximately four weight16KiB stages and four activation1KiB stages plus scales/barriers. There is no candidate profile proving that four-stage starvation is the limiting cost, and extra allocation can reduce residency. Consequently, eight stages is not automatically a bandwidth improvement and is not the selected next experiment.

Register correction remains important: vendor SASS actually contains `USETMAXREG` instructions and a REG128 header; the TS CUBIN has REG32/31 and no emitted register-reconfiguration instructions despite PTX160/48 hints. Source hint arithmetic cannot establish a53,248-register TS allocation or one-CTA occupancy. Actual occupancy has not been measured for the new candidates.

## Weight reuse is not a supported large savings claim

The older NCU three-layer sample observed FC1/FC2 DRAM reads only1.00052–1.00138 times the logical active-expert FP8 weight bytes. DRAM utilization was56.36–75.54% of peak, whereas tensor-pipe active was3.46–4.48%. Thus there is substantial execution time in memory supply, but no evidence that repeatedly fetching whole expert weights is the main waste. Cache behavior already recovers repeated token-tile accesses. Avoiding the FC1/activation global intermediate traffic removes only about0.8–1.0% of active weight bytes; the failed resident implementation confirms that eliminating those buffers is insufficient by itself.

The current real oracle routing has206/159/115 active experts and208/168/146 token8 tiles in layers0/19/39. This is different from the old NCU sample's204/163/116 experts. Native and TS global kernels preserve8 FC1 output tiles and16 FC2 output tiles per token tile; the resident cluster instead packs those into four CTAs with two FC1 and four FC2 stage invocations each. Its slower complete result does not isolate whether serialization, cluster residency, six setup/drain sequences, PDL or another factor dominates. No new kernel profile was collected, so none of these can be labeled the measured cause.

A conditional bandwidth calculation using the older six NCU launches yields19–30% GEMM-time reduction if measured traffic could run at90% peak. This is an ambitious accounting scenario, not achievable headroom demonstrated by either new implementation. It cannot be subtracted from current unprofiled complete-call medians or multiplied across40 layers.

## Selected experiment: M128 PDL false to true

The existing API supports the complete chain:

- `config_mapper.py:1060` supplies `use_pdl=1` and `do_pdl_wait_for_num_non_exiting_ctas=1`.
- `batched_gemm_kernel.py:2417` performs `griddepcontrol.wait` before dynamic metadata access; persistent tasks then use their normal routing ownership.
- `batched_gemm_kernel.py:3326` launches with `use_pdl=True`.
- `moe/runner.py:2362,2384` passes the same enable flag to the original native activation and finalizer.
- Native activation (`data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:247`) triggers dependent launch and then waits before reading prior outputs.

The new CPU AOT confirms each candidate GEMM has one `griddepcontrol.wait` and two `launch_dependents`; actual SASS adds one `ACQBULK` and two `PREEXIT` instructions. Generated host code stores launch attribute ID6 with value1. Shared allocations and no-spill findings remain unchanged. The runtime capture must additionally show actual programmatic edges; source flags alone are insufficient.

This is a conservative existing dependency schedule. Entry waits limit prefetch overlap, and release after a work loop is not full FC1/FC2 compute overlap. A suggested3–5µs budget is unmeasured: relative to complete native calls of approximately81–121µs, even that would be only2.5–6.1%, not automatically10%. Prior M128 gains and a hypothetical PDL gain must not be added across experiments.

The frozen gate is all15 original logical intermediate/finalized values exact under independent live maps, changed inputs through reused graphs, guards/input/weights/RNG unchanged, CPU/GPU CUBIN identity, and a fixed-nonzero-grid two-GEMM empty `ncta=padded=0` graph followed by restoration. Empty activation/finalizer semantics are deliberately excluded. CLC invalid tiles must skip and drain; CTA exit is also an implicit dependency release. Graph nodes/edges are queried without mutation.

Then one AB round of100 complete calls for each of the three real cases must each beat native by5% before any three-round claim is attempted. If that passes, run all15 in AB/BA/AB order,100 calls each, requiring each real median>=5% faster and no stress regression>5%. The vendor baseline retains PDL=True throughout. No extra PDL=False timing arm, runtime promotion or service claim is authorized by component correctness alone.
