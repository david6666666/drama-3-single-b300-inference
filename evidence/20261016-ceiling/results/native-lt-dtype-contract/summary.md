# Native BF16 Lt dtype-contract diagnostic

The first strict capability filter selected no algorithms. Its complete sequence
both arms remained F.linear: the 0.024148 ms difference is an A/A observation,
not an optimization. Original results and their STOPPED label remain unchanged;
no candidate numeric or timing claim follows from that run.

Both first heuristic descriptors request BF16 A/B/D (14), FP32 compute (68), and
FP32 scale (0). Actual getters confirm those values. The capability flag is
0x10202: documented INPUT_16F plus ACCUMULATOR_32F, whereas INPUT_16BF would be
0x20000. This contradiction remains explicit; the flag is not redefined.

Eight full-output diagnostic cases pass byte equality against rounded FP64:
large X/tiny W (powers +80/-80), the reverse, cancellation, and accumulation
that retains small summands. This rejects ordinary unscaled FP16 narrowing for
these inputs. It does not prove all inputs or all internal instructions native,
and does not authorize a general capability-filter exception.

The initial CPU-only C++ query, without a CUDA context, returned status7 and is
preserved. The first context-initialized C++ probe loaded system cuBLAS13.7 due
to inherited LD_LIBRARY_PATH overriding RUNPATH. A follow-up pinned only that
child's library path and reproduced flags0x10202 on cuBLAS13.1.1. The Python
Torch and absolute ctypes paths used venv13.1.1 throughout, confirmed by an
independent /proc/self/maps probe without CUDA initialization. Neither Python
numerical result was obtained with the system13.7 library. The corrected C++
queries and original cross-version observations are both retained.

No GPU service measurement or runtime change occurred. Probe ownership was
checked only before/after, not continuously; initial full model feasibility
used a sampled2s watchdog. The framework target and original quality limits
remain unchanged.

Official numerical-flag semantics:
https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-numerical-impl-flags-t
