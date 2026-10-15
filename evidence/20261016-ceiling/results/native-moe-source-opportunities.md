# Native MoE opportunities, CPU source audit

No GPU run or speedup claim accompanies this audit. Full top8, checkpoint FP8 expert scales, BF16 dense and state computation remain fixed.

## Reserved SMs do not constrain expert GEMMs

Installed FlashInfer 0.7.0.post1 reads `FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS` (default 8) in `flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh:49-104`. Its callsites size only cooperative routing/permutation grids, not FC1/FC2. The real B64 trace uses BlockScoresKernel and ClusterKernel, bypassing that cooperative branch. Current mixed cooperative permutation occupies 0.23855 ms per graph: even removing that entire kernel cannot recover 0.85% of a 28.32 ms mixed graph. No reserve sweep is justified.

## PDL waiting is not removable activation computation

From the real traces, per-graph sums of each target kernel's overlap with its immediately preceding FC producer and the tail after producer completion are:

| Shape | Kernel | Total ms | FC overlap ms | Tail ms |
|---|---|---:|---:|---:|
| B64 | activationDeepSeekKernel | 0.356259 | 0.218756 | 0.137503 |
| B64 | finalizeKernel | 0.323117 | 0.194145 | 0.128972 |
| D64 P1280 | activationDeepSeekKernel | 1.474823 | 0.192982 | 1.281841 |
| D64 P1280 | finalizeVecLoad | 0.503907 | 0.184974 | 0.318934 |

These are local trace-tail budgets, not a complete critical-path proof. Sources: `../drama3-scale150-20261015/native-bf16-projections-b64/trace.json.gz` and `mixed-profile/trace.json.gz` (three graphs each).

## Concrete bounded candidates

1. Combine the native shared-expert gate/add with next residual RMSNorm. Preserve BF16 sigmoid, weighted shared output, routed+shared sum, residual sum, and existing RMS reduction geometry. Existing combine costs 0.0656 ms/B64 and 0.2632 ms/mixed; the 40 next norm kernels add approximately 0.099 and 0.164 ms. Actual savings must be lower than these combined budgets because arithmetic and input/output traffic remain. A default-off implementation can avoid 40 launches and one MoE intermediate allocation without changing expert GEMMs. Existing shared-stream join must remain before returning output parts. This is the current authorized feasibility experiment.

2. Native activation scheduling: installed `trtllm_fused_moe_dev_kernel.cu:235-417` uses 128 threads/CTA, scalar element accesses and CTA reductions across 128-channel FP8 groups. For M1344/top8 this launches 10,752 CTAs (four tokens per CTA). A warp-per-row implementation with four channels/lane could preserve FC1 FP8 dequantization, gated SiLU, exact 128-element amax, clamp/division and FP8 boundaries while replacing CTA barriers with warp reductions and vectorized accesses. FC1 intermediates must first be captured through isolated FI instrumentation; public `do_finalize=False` returns FC2, not FC1. Even deleting the entire exposed activation tail would recover only about 0.48 s in the measured 1622-decode/197-mixed ledger, roughly 2.1% of 23.23 s. No implementation or GPU test approved yet.

3. FI `do_finalize=False` exposes FC2, expert weights and permutation map, so native finalization could eventually fuse with shared combine/residual RMSNorm. Must retain the ordered top8 FP32 accumulation, BF16 cast, subsequent BF16 boundaries, and PDL producer synchronization. Pure/mixed whole-tail+combine+norm budgets are approximately 0.294/0.746 ms; genuine saving is substantially lower. More integration risk than item 1.

Fused shared SiLU plus BF16 down GEMM has only approximately 0.20 ms total pure-B64 component budget, with the removable SiLU near 0.07 ms. FI's fused shared expert support uses the routed FP8 tensors and would require a prohibited new quantization of the BF16 shared expert. The prior 144 native dense tile trials and wider-N/blocked-layout/PDL experiments are not candidates for a repeated sweep.

## Goal scope

The rich service ledger (source ab6da532083854670348583d9ec9f2a16d184dfb) is 4373.008671 fps, 23.2336608 s; 6765.407263 fps needs about 15.0177212 s, a reduction of 8.2159396 s. None of the identified native MoE candidates has evidence for that missing budget. The counters already show approximately one logical weight read and 4.3-5.8 TB/s at B64, so low tensor utilization alone is not evidence of a large compute optimization. This is not an impossibility proof. Service scheduling/codec interaction and measured padding waste remain independent opportunities; old isolated or synchronous timings are not current service lower bounds.
