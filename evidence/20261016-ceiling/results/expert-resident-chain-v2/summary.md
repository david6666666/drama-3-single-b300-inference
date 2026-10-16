# Complete single-expert resident chain: exact PASS

The frozen v2 gate passed 489/489 exact checks on GPU2. One original layer-0 expert (205) processes its eight original routes, followed by effective row counts7 through0. This is a complete resident FC1 → native shared activation → DSM gather → FC2 chain for one selected expert, not a complete256-expert implementation or a performance result.

All native E4M3 FC1 bytes, FP32 FC1 scales, E4M3 activation bytes, FP32 activation scales, and BF16 FC2 values match the original native full-call oracle. Eager execution, graph replay, inactive output sentinels, all input/weight/metadata tensors, and RNG pass. The original installed ordered top8 finalizer also matches after replacing only tested branches; all other expert branches use archived native outputs, so this finalizer check is explicitly oracle-assisted.

The CPU and GPU CUBIN SHA256 is9eb4c4f83850b3743118c50628631dfdcf2707fb607306c377ad9a743ceab0b1. Actual launch shared memory is78,464 dynamic plus1,024 static =79,488 bytes per CTA. Stack/local spill is zero. The 53,248-register figure is only a source-hint budget, not actual allocation: emitted SASS has no SETMAXNREG, highest R29, and CUBIN REG32. No CTA-residency conclusion is established. The four-CTA cluster retains original group1 MMA. TMEM32 is allocated and relinquished once, lent to six drained stages, and deallocated once after the final stage.

The original first attempt in../expert-resident-chain is retained as a host-ABI failure: it passed compilation/input checks but used compile-time arguments at runtime, before any candidate kernel launched. V2 changes only the runtime invocation to the installed native _launch_arg_tuple, omitting constexpr configs; its kernel CUBIN is unchanged. No numerical tolerance or native arithmetic was changed.

No timing was performed. Full256 native metadata, changing route distributions, shared-row offsets, padded scale strides, and a later zero-debug full-MoE timing gate remain separate work in../expert-resident-full.


Register-accounting correction: the 160/48 PTX/source role hints are not equivalent to actual SASS allocation. For the full debug and zero-debug resident CUBINs, cuobjdump reports REG32, emitted SASS has no SETMAXNREG, and the highest general register is R29. Any earlier 53,248-register/CTA or one-CTA-per-SM claim is an unproven source-budget inference and must not be used as occupancy evidence. Shared-memory totals and zero-spill checks remain valid; actual occupancy requires its own measured/compiler evidence. See ../resident-register-accounting-erratum.json.
