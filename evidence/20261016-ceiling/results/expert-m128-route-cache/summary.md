# FC1 scale-route caching: exact pass, local continuation fail

All 1,516 exact checks pass, including the same 15 immutable native oracles, eager and reused graphs with changed routing maps, FP8 bytes/FP32 scales/BF16 FC2/native ordered finalizer, padding/guards/input/weights/RNG and same two-GEMM graph 0→positive→0 restoration. The old metadata-ready candidate matched these same bytes, so this establishes their common exact numerical contract without output-value reordering or relaxed tolerances.

| Real layer | Vendor complete call (µs) | Candidate complete call (µs) | Latency reduction | ≥5% gate |
|---|---:|---:|---:|---|
| 0 | 120.871 | 117.194 | 3.042% | FAIL |
| 19 | 99.385 | 96.603 | 2.800% | FAIL |
| 39 | 80.055 | 76.704 | 4.186% | FAIL |

This is one AB screening process on GPU2, five warmups and 100 graph calls per arm. Each arm includes live metadata, FC1, native activation, FC2 and native ordered-top8 finalization; common input quantization and logits-to-top8 are excluded. Physical B64 means 63 live rows plus original padding. All three real cases miss the predeclared 5% continuation gate. Twelve stress performance cases and three repeated rounds were not run. There is no runtime, full-model or service promotion, and no all-concurrency target claim. Differences against the older candidate's separate run are not a causal speedup measurement.

The only kernel change caches each valid scale-producer lane's physical route row once per active work tile, after the original PDL wait. It removes the same-address route LDG from 16 K128 iterations while retaining the original scale cp.async, masks, stage acquire/commit, FP8/BF16 boundaries and FMA order. CPU proof sees seven static unrolled K-body LDGs become zero, one extra prepare LDG, and total emitted SASS LDGs20→14. That proves the dependency was removed; it does not prove it dominated aggregate long-scoreboard stalls or service latency.

FC1 CUBIN is `06974be36c9213567e0f77dd382ef6128caf806cb601f83d97dc4b8fe69f3475`; FC2 remains `ea52321ec64fc9976a306b7c7aeeb0805033c2e024034e5cc6ce266e7d95b170`. Actual FC1/FC2 shared allocations are74,616/74,664 bytes, actual base register headers remain32, and stack/local/spills are zero. Host launch IR is identical after excluding only the embedded changed FC1 binary constant. Source PTX160/48 register hints are not an occupancy proof. All158 frozen files remain byte-identical after execution.

The initial CPU whole-host-text equality assertion failed because host IR embeds the changed CUBIN; its script/result/log are retained in `cpu-initial-host-text-check-failure`. Only that checker excluded the embedded binary data line; kernel, numeric checks and performance thresholds did not change. Broad workspace lint reports style/DSL idioms in preserved upstream-style code; AST and isolated syntax/undefined-control-flow lint pass. No automatic DSL rewrite was applied.

`protocol.json`, `frozen-inputs.json`, `cpu-route-cache-proof.json`, `driver-prepare.json`, `gate-result.json`, `feasibility-result.json`, `launch-result.json` and process audits retain the complete evidence. Pre/post GPU queries are empty and sampled ownership has no foreign PID; these are sampled checks, not continuous exclusivity proof.
