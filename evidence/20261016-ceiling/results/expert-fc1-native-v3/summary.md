# Native FC1 arithmetic gate v3

The expanded gate passed 438/438 exact checks. This is an arithmetic and state-validation result, with no timing or service-performance claim.

The 15 original routing fixtures cover layers 0, 19, and 39 with real, reversed-row, left-only, right-only, and worst-global-288 distributions. All 256 original experts, original top-8 routing, 0–64 routes per expert, partial tiles, padding, eager execution, graph reuse, inputs/weights, and RNG were checked. Candidate E4M3 bytes and FP32 output-scale bits match the installed native FC1 oracle.

Explicit synthetic cases also passed for zero input and scaled fixture input yielding amax approximately 1e-14 and 1e-10. They change only input scales, preserving original checkpoint weight bytes/scales and fixture routing; they are not real service fixtures. The native contract uses 448*rcp.approx.ftz(amax) followed by its approximate reciprocal, with zero handled separately. The non-native Prims 1e-12 floor was removed. Earlier failed v1, expanded-zero, and v2 artifacts remain preserved.

The CPU/GPU CUBIN hash is 476495717a31847f92b368bd263d26855b4d9cdec71451466653ced2e4fe3ee8. Actual shared memory is 73,592 bytes; stack and local spills are zero. Dynamic register allocation remains 8 epilogue warps at 160 plus 8 other warps at 48, or 53,248 registers per CTA; the static REG32 header does not imply two CTAs per SM.

The next gate is the complete resident FC1 → native activation → FC2 chain. Component exact passes do not establish full-chain correctness or net performance.
