# Native activation shared-memory contract

This is an artifact-only arithmetic/interface prototype, not a production or speed result. The subsequent shared-body GPU gate passed; see results.md. The already-tested original host activation launcher is not being rebenchmarked. The candidate moves actual input/output storage to shared memory and changes thread ownership from native 128-thread block reduction to one complete warp per token. It preserves both independent FP8 boundaries and all scalar activation arithmetic.

## Proven CPU/source contract

Original code: `flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu`, `activationDeepSeekKernel` at lines236–374, `silu` at55, `Float2Max` at149. The exported host helper is `trtllm_moe_run_deepseek_fp8_activation` in `data/csrc/trtllm_fused_moe_kernel_launcher.cu:3067`. It launches a separate global kernel, so it cannot itself execute inside a resident FC1→activation→FC2 CTA.

Native logical FC1 is `[route,1024]` E4M3, with **up512 before gate512**. Its global physical scales are block-major `[8,total_padded]`; oracle exports transpose to `[route,8]`. For rank r=0..3, input channels are up`[128r:128(r+1)]` and gate`[512+128r:512+128(r+1)]`. These retain separate FC1 scales and E4M3 roundings. Do not substitute FP32 pre-quantization accumulators or BF16 values.

For one 128-channel block and token:

1. `x1 = scale_up * fp32(up_e4m3)`; `x2 = scale_gate * fp32(gate_e4m3)`.
2. `out = (x2 / (1 + expf(-x2))) * x1`, with native `-use_fast_math` and FTZ behavior, no alpha/beta/clamp controls in Drama3.
3. `amax = max(abs(out[0:128]))`; activation scale is `max(amax / 448, FLT_MIN)`.
4. Each output is `cutlass::float_e4m3_t(out / activation_scale)` with native round-to-nearest and finite saturation.

This activation scale is **not** the FC1 quantizer's `rcp.approx(448 * rcp.approx(amax))` scale. The native SASS shows `MUFU.EX2`, `MUFU.RCP`, `FMUL.FTZ`, multiplier `0.0022321429569274187` (= FP32 1/448), floor `1.1754943508222875e-38`, and `F2FP.SATFINITE.E4M3.F32`. Reusing a generic normalized activation or quantizer would not establish equivalence.

The CPU scan covers all15 native oracles, 512 expanded routes each. It finds no FC1 or activation NaN bytes, finite positive scales, **22 FC1 negative-zero bytes and149 activation negative-zero bytes**. Therefore comparisons must inspect bytes and FP32 scale bits, not numerical equality alone. Candidate max uses only finite nonnegative values on this domain, so unlike a sum its regrouping introduces no rounding. All-zero, signed-zero and NaN behavior should still receive a separate synthetic test against the native operation. Such cases are diagnostic extensions, not substitutes for the real oracle gate.

## Resident interface and lifecycle

The shared device helper in [shared_activation.cu](shared_activation.cu) accepts:

- FC1 bytes `[8,256]`: each row up128 followed by gate128;2112B including scales.
- FC1 scales `[2,8]`: first up, then gate, indexed by token.
- Activation destination `[8,128]` bytes plus `[8]` FP32 scales;1056B.
- A uniform valid-row count0..8. Each valid token owns exactly one complete warp.

The FC1 agent confirms current compiled FC1 CTA has512 threads and8 epilogue warps. Only warps0..7 execute activation; the other warps do not return from the enclosing kernel and join outer synchronization. Each active lane handles four contiguous channels. Five warp shuffles reduce128 maxima; there is no per-token CTA synchronization. The caller first completes FC1 producer/MMA/epilogue and publishes the shared data. After activation it publishes the rank fragment before cluster consumers read it. Four rank fragments assemble token-major `[8,512]` and block-major scales `[4,8]`, matching FC2's tested logical input ABI. A consumer-completion cluster barrier must finish before producer storage is reused. This prototype does not yet test DSM, shared swizzles, TMEM-to-shared epilogue scattering, or full-cluster lifetime.

The diagnostic wrapper genuinely stages FC1 into shared memory, calls the shared helper, and exports shared activation for byte comparison. It uses two CTA publication barriers and128 bytes of shared canaries. No repeated original host activation gate is proposed for the15 oracles.

## Offline resource result and proposed gate

AOT CUDA13 compiler command uses `-O3 -use_fast_math`, `sm_103a`, and the same installed CUTLASS E4M3 converter. It reports **32 registers/thread, zero stack, zero spill loads/stores**, and3296B user shared memory. `cuobjdump` reports4320B shared for the compiled function; both tool outputs are retained rather than silently treating these as identical measures. Logical input/output fragments total3168B; the128B canary area is diagnostic-only. This standalone resource count does not predict the eventual resident kernel's combined register lifetime or occupancy.

Prepared [inputs.pt](inputs.pt) contains60 rank fragments (15oracles×4ranks) plus4 partial cases cycling valid-row counts0..8 across token8 groups. [cpu-contract.json](cpu-contract.json) records all input/source/object hashes, layout, finite/signed-zero statistics and `cuda_initialized=false`.

The completed one-shot GPU gate was exact eager and captured-graph comparison of all native activation bytes/scales, partial ownership/canaries, untouched inputs/scales/counts/RNG, plus explicitly separate zero/NaN diagnostics. Stop on the first mismatch and preserve it. No timing, service launch, full-model computation change, or acceptance promotion is included. Root explicitly approved the new shared-body gate before its launch; no timing or service continuation is authorized by this result.
