# CuTe shared activation numerical gate

**PASS: 137 cases, 2066 exact checks.** The new CuTe/inline-PTX helper has its own result; the prior C++ 137-case/2063-check pass remains unchanged. The three additional checks are frozen-file hash checks, not additional numerical cases.

The exact gates cover 15 archived native oracles × four rank fragments in eager and reused CUDA graph execution; row counts 0..8; native zero, signed-zero, FLT_MIN/subnormal, NaN, and saturated-E4M3 synthetic references; every output byte and FP32 scale bit; shared/global guards; all inputs, row counts, and CUDA RNG. Native synthetic references were read from the original frozen C++ experiment; no original host kernel was rerun. Source helper SHA `f73a86fbfde68be1592f4d0bbfc52756081a3c78be6583b2f4090a3ccb49542d`.

CPU-only AOT passed without CUDA initialization before launch. The driver froze hashes and clean provenance SHA 95fcb0e, checked local storage and idle GPU6, and ran once without timing. PID 585933 exited 0; the final compute-app list is empty. No source fixes or retries were needed. Runtime source was untouched.

This proves the standalone shared-memory arithmetic/layout helper under its declared 512-thread interface. It does not validate the future four-CTA resident chain, DSM handoff, FC2 operand swizzle, combined resource occupancy, or performance.
