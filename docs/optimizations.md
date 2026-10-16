# Runtime optimizations

Features in the source are not all enabled by the bare serve command. The table
distinguishes the core implementation from explicit experimental profiles;
local kernel improvements are not counted as service improvements.

| Optimization | How it reduces work or overhead | Current qualification |
|---|---|---|
| Persistent slots and GPU request state | Keep KV/GDN state, generated codes, positions and repetition history on-device; manage slot reuse explicitly | Core implementation; partial-prefill primary/suffix padding isolation is fixed and checked |
| CUDA Graph batching | Capture fixed batch/shape buckets and replay decode, prefill and codec operations | Core implementation; startup includes capture and warmup |
| Packed projections and native MoE dispatch | Combine compatible QKV/GDN projections; use local small-batch kernels and FlashInfer large-batch kernels while retaining original top-8 experts | Core paths; packed router/shared projection and fused input/residual require explicit flags |
| Packed and mixed prefill | Pack prompt rows and share backbone projections/MoE with active decoding instead of always interrupting it with a separate padded pass | Opt-in; complete single-pair quality evidence, no latest all-concurrency acceptance |
| Compact GDN and padding work | Compact active state handling and avoid unnecessary work on reserved padding slots | Opt-in in the later C64 profile; full state/ownership gates retained |
| Streaming DAC and codec overlap | Preserve convolution/attention history and decode only new audio frames; optionally overlap codec and model work with private capture workspaces and queued PCM delivery | Streaming codec is core; asynchronous execution is opt-in; shared-workspace failure remains documented |
| Two-frame pipeline and deferred admission | Start the next backbone once semantic/codebook-1 dependencies are ready; permit pairing while arrivals are below admission thresholds | Opt-in; complete quality in single C64 trials; no all-concurrency speedup claim |
| Exact causal-prefix MoE row reuse | Compute identical prompt-prefix rows once and gather their results through an inverse map | **Experimental, default off:** 9974 correctness checks pass; local graph latency improves 6.23–6.77%, but the complete service pair is slower |

Resident expert fusion, M128/scale-cache variants, transparent compressed
allocation and codec cached-linear variants did not clear their declared
performance gates. Dense FP8 serving is rejected. Those experiments are retained
for auditability and are not recommended serving optimizations. The CPU SIMT
proposal was stopped before GPU execution; it is not a measured speedup.

High-concurrency profiling shows that a faster isolated decode graph does not
ensure higher service throughput: prompt work, codec contention, graph padding,
admission and final drain also count. Sampled native MoE kernels move substantial
expert-weight data while showing low tensor-pipe activity. Higher CTA residency
alone did not improve issue progress enough. Neither an isolated 7734 frames/s
calculation nor a bandwidth estimate establishes an end-to-end upper bound.
See [scaling analysis](performance-gap.md) and
[the complete experiment ledger](scale150-progress.md).

