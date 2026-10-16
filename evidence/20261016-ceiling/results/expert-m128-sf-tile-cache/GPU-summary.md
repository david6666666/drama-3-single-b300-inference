# Full SF cache: exact PASS, continuation FAIL

The original fifteen native fixtures, all FP8/FP32/BF16 boundaries and ordered-top8 final outputs passed under eager and reused-graph execution. The GPU gate passed **1518/1518 checks**, including token/expert mapping ownership, weight/input/RNG preservation, output guards/padding, actual metadata-producer full dependencies, CPU/GPU CUBIN identity, and fixed-grid same-graph empty→positive→empty restoration. Empty execution includes only FC1/FC2; it makes no unsupported empty activation/finalizer claim.

The predeclared single complete-call AB100 screen failed the >=5% requirement in every real layer. Post-timing verification passed another **148/148 checks**. No three-round measurement, stress timing, full-model/service run or runtime promotion followed.

| Real fixture | Native µs/call | Candidate µs/call | Latency reduction |
|---|---:|---:|---:|
| L0 | 121.043844 | 116.867199 | 3.45052% |
| L19 | 100.843201 | 96.635199 | 4.17282% |
| L39 | 80.354881 | 76.799998 | 4.42398% |

These are one A/B observation with five warmups and100 replays per arm, not statistically established gains. The measured graph includes live native routing metadata, FC1, native activation, FC2, native ordered finalization and the required metadata-ready graph edges. Common input quantization and logits-to-top8 selection are excluded equally. There are no extra debug export kernels in the timed graph. The native baseline retains tactic(8,5) and PDL. The candidate uses original FP32 scale bits, the new576B producer cache/AsyncThread publication and the unchanged metadata-ready FC2.

Relative to the prior metadata-ready candidate the SF producer is the isolated code change; relative to the vendor baseline this is a complete implementation comparison. Independent earlier timings cannot be subtracted to assign the small difference to the scale cache. The unchanged original negative experiments remain authoritative.

Actual GPU-compiled FC1 is `311a2b1d147155ecebbad21e522cfeeba3c75ebe4b30ada8bb5eca98308d8d33`, total shared75,192B; FC2 is `ea52321ec64fc9976a306b7c7aeeb0805033c2e024034e5cc6ce266e7d95b170`, shared74,664B. Both match their CPU AOT binaries, report REG32 and no stack/local/spill. No source-register-hint occupancy inference is made. GPU-generated host IR and resource reports are retained for both stages.

The actual loaded native binary is `/home/zjy/.cache/flashinfer/0.7.0.post1/103a/cached_ops/fused_moe_trtllm_sm100/fused_moe_trtllm_sm100.so`, SHA256 `25899ed3f56e3b55cc3c4bf790cfca080a7c4c598e61927affc1279b30266de7`. The loaded CUTLASS overlay is4.7.1; the separate installed distribution metadata says4.6.2. Both path/version fields are recorded rather than conflated.

GPU4 UUID `GPU-4ae7e013-1aac-31d8-8b8e-c27530f1c6ed` was empty before and after both owned processes. The launcher sampled ownership every2seconds; this is not continuous exclusivity proof. Gate and feasibility process elapsed38.074s/29.603s include source/checkpoint hashing, loading, compilation and correctness work, and are not performance metrics. Frozen inputs total14.736GB. No clocks/cache policy were changed.

Protocol SHA256 `53cad5316ad86d1e4f918de653d10599495d57e7cfe95de9f20076c298150820`; full202-file manifest SHA256 `d8bfc3c7f02732f62faedb019030ada212223e997de7f9487e6b20bb30d51a11`. The earlier CPU-only summary and manifest are preserved; the host-only GPU4 adaptation is recorded with the original CPU driver snapshot. This report closes this bounded experiment with a negative continuation result.
