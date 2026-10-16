# Independent CPU review: fixed8-lane grouped SIMT

This reviews the current artifact only. No GPU execution, runtime change or numerical acceptance is implied.

## Numerical contract

The source decodes original E4M3 bytes through FP16 into FP32. Every finite E4M3 value is exactly representable in FP16 and FP32; this is exact decoding, not a new FP16 model precision. The current inner lane recurrence uses explicit FP32 RN FMA, then shuffles4/2/1 within8lanes. For each128-element K block it forms one FP32 RN activation-scale×weight-scale product and applies `fma.rn.ftz.f32` to the cumulative FP32 output. FC2 ends with RN BF16 conversion. This matches the native placement of K128 scales and the native outer FTZ contract, but it does **not** establish equality to the internal K32 UMMA reduction tree.

Installed FlashInfer `prims_ts/batched_gemm/smem_deepseek_sf_resources.py:352` describes activation×weight scale formation; `accumulate_scaled_t2r` at447 explicitly uses packed RN FMA with FTZ. Current artifact SASS has unscaled FFMA without FTZ, FMUL and outer FFMA.FTZ. Do not dequantize each element before the dot product, merge scale blocks, apply route weights inside FC2, or use BF16 accumulators.

The host contract must restrict rows to0/1/2 and outputs to the fixed1024/2048 dimensions. Under these restrictions, every participating warp has all32lanes active at the full-mask shuffle:0rows returns uniformly; the output dimensions are exact multiples of16columns/CTA. This does not make the helper safe for arbitrary output tails or rows>2.

FC1 currently emits raw FP32 diagnostics only. A complete native path must still preserve independent up/gate K128 output-block amax, original FP8 conversion and dequantization scales, then native activation. The FC1 quantizer uses `q=448*rcp.approx(amax)`, `dq=rcp.approx(q)` with the native zero case; activation has a different scale construction (`max(amax/448,FLT_MIN)`) and native fast-math SiLU. These cannot be substituted by one generic quantizer. The previously exact activation helper consumes quantized FC1 values and scales; it cannot accept prequantized rawFC1 and silently skip that boundary.

## What existing oracles prove

The prior one-expert FC2 CPU diagnostic showed that all16,384 original BF16 values matched FP64-rounded output for that finite real fixture. The full15-oracle GPU evidence separately validated the native M64 implementation and layout. Neither proves FP32 sum-tree interchangeability for a new SIMT implementation. BF16 agreement can hide FP32 partial differences, and final BF16 disagreement alone does not locate the cause.

For the proposed3real×2route×64output CPU subset, compare original decoded input bytes and original scale values first, then independently track all four unscaled K128 partial sums, scaled FP32 accumulated totals, and final BF16 values. Use the same native scale indexing and expanded/permuted route identity; never match outputs by sorting. An independently computed FP64 reference checks layout/sign/scale magnitude, while a serial-FP32 and explicit8lane-tree pair isolates the candidate's own reassociation sensitivity. Because native unscaled UMMA partials were not archived, disagreement with the native final value cannot by itself prove that the tree is the only cause. Preserve it as a diagnostic failure or uncertainty, not a passed precision gate.

A libc `fmaf` CPU model needs explicit native outer FTZ emulation (flush subnormal operands and result with signed-zero behavior), or a measured real-domain proof that none enters that path. Ordinary Python multiply/add is not fused. Signedzero, NaN, amax/scale-floor and FP8converter cases remain separate contracts; real finite samples do not cover them.

## Optimistic instruction budget

For40layers,64tokens,fulltop8, FC1(2048×1024)+FC2(512×2048), the workload is64,424,509,440 MAC. One full warp K128 iteration with two routes covers4outputs×2routes×128=1024MAC. Even counting only the stated24E4M3 pair conversions+48half-to-float instructions+32FFMA gives104warp instructions/1024MAC, hence6,543,114,240warp instructions. Counting152 for the complete steady two-row loop gives9,563,013,120.

Under an explicitly optimistic model of148SMs,4warp instruction issues/SM/cycle and2.05GHz, these are5.3915ms and7.8799ms respectively, before routing, scale/activation/finalizer overhead, load stalls, tail inefficiency or occupancy losses. This is an issue-budget model conditional on the stated dispatch/clock assumptions, not a hardware impossibility proof. It already makes this fixed implementation a poor route to the approximately4ms combined GEMM budget.32registers/0spill does not offset the instruction count or imply measured occupancy/throughput. Avoid a GPU run merely to confirm a weak arithmetic-throughput premise; no arbitrary tile sweep follows.

## Source fingerprints

- `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/native-fp8-simt-cpu-audit/fixed_subwarp8.cu`: `fbadcc85fec2387cf24f0091cb839a9c7080d0d608580512141653a16e28341b`
- `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/native-fp8-simt-cpu-audit/kernel.sass`: `bf8a4d284149e958cfe6f1e55d66647ff23ac367f5d3047a9922b4258e771d5f`
- `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/native-fp8-simt-cpu-audit/resource-usage.txt`: `d62a43847fea35b0267a21de852954adaaf2621b09fefe56a65a5bd6392d9998`
- `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/expert-fc2-native/independent_cpu_layout.py`: `ca7756371033af16e0149f0a9eec47b23d73b004cb1e0c4e54a651237e8bbe5d`
- `/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016/expert-activation-native/shared_activation.cu`: `84fbc06ed358c794dceb59e7c0197873a798fe132fd8bf52b0ef4f2d07f9a46d`
