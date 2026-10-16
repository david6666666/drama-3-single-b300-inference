# FC1 scale-route cache: predeclared artifact-only candidate

Parent: frozen metadata-ready complete M128 implementation, source 95fcb0e.
Only FC1's per-work-tile scale producer caches each valid lane's original physical route row before the K loop, replacing 16 identical route-map loads with one. The original predicate, per-K scale copies, stage acquire/commit, FMA order, native FP8/BF16 boundaries, PDL and metadata-ready FC2 graph are preserved. FC2 uses its unchanged source/CUBIN. No installed/runtime source changes.

The aggregate long-scoreboard observation is a hypothesis generator, not proof this address dependency is the dominant cause. Logical valid route-load bytes drop by 245,760 bytes per 512-route fixture, which is tiny relative to expert weights; expected mechanism is fewer repeated dependent loads, not fewer weight bytes. No numerical relaxation or retrospective promotion of prior failed experiments.

CPU prerequisites: actual compiled PTX/SASS removes route reads from the K loop, one refreshed route per active work tile; no stack/local/spill, no increased shared memory or compiled residency resource loss; unchanged FC2 CUBIN. Preserve any compile or exact failure.

GPU gate only after CPU freeze/review: all 15 native logical oracles eager/reused graph with changing maps, ownership/input/weights/RNG/guards, exact FP8/scales/BF16/finalizer, actual metadata-ready graph edges; separate 0→positive→0 two-GEMM same-graph empty gate. Native baseline retains production PDL. One AB screen for the three real layers includes live metadata + all MoE body + native finalize; common input quantization and logits-to-top8 excluded. Each real complete-call reduction must be at least 5% before any repeated trial; a screen is not a formal or service gain. No GPU run authorized by this file alone.
