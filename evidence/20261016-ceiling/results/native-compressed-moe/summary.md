# Transparent compressed VMM: native MoE exact PASS, performance screen FAIL

The complete native MoE arithmetic is unchanged in both arms. All 3417 checks passed, including actual compression allocation flags, full original weight bytes, the fifteen native fixtures, live routing ownership, every original FC1/activation/FC2 boundary, ordered-top8 final output, eager and reused graphs, input/weight/RNG preservation, guards and cleanup. The single predeclared A/B screen did not meet the continuation gate. No additional rounds, full-model, service or quality run was performed.

| Scope | VMM NONE µs | VMM GENERIC µs | Latency reduction |
|---|---:|---:|---:|
| L0 | 120.874557 | 120.099201 | 0.6415% |
| L19 | 100.473280 | 100.696316 | -0.2220% |
| L39 | 80.344324 | 80.093756 | 0.3119% |
| L0→L19→L39 sequence | 295.408325 | 295.115204 | 0.0992% |

The gate was frozen before GPU execution: the three-real-layer sequence must reduce latency by at least 5%, while no individual layer may regress by more than 1%. The sequence improved by only 0.099226%, so the primary gate failed even though the per-layer non-regression condition passed. These are one A/B observation with 100 replays each, not a statistically established 0.10% improvement.

## Boundary and controls

Source is `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`. Both arms use the same installed original native DABodyRunner, tactic `(8,5)`, and PDL enabled. Timing includes live native metadata population, native FC1, native activation, native FC2 and native finalization. Shared input quantization and logits routing are excluded equally. Original q/IDs/routing weights, all 256 experts and eight selected experts, original FP8 weight bytes, FP32 scales and all intermediate precision boundaries are preserved. No Prims/CuTe replacement kernel is imported.

Only the allocations backing original contiguous w1 and w2 differ: VMM NONE versus VMM GENERIC. Scales remain the same original read-only tensors. The original native intermediate buffers retain their dtype/shape/stride; the first five workspace fields and output have equally applied guarded views with an asserted 256-byte alignment. Native scratch fields retain the original preparation allocation and initialization.

The three-layer sequence is one graph containing three complete native MoE calls in order on their own fixed real inputs and private metadata/workspaces. It is not a three-layer neural network with actual intermediate backbone dependencies, and it is not a service benchmark. Both arms have the same graph boundary and capture policy. Individual graphs are reused across all five routing distributions per layer before timing. The final sequence outputs, maps and guards are checked after timing without first overwriting them through another replay.

## Allocation and ownership evidence

All twelve VMM weight allocations reported the expected actual flag (six NONE and six GENERIC). They requested 4831838208 bytes and mapped 4831838208 bytes in total. These sizes describe virtual allocation capacity, not physical compressed traffic or compression ratio. All original weight bytes matched before native use and after the gate/timing. The actual loaded native `.so` path and hash are recorded in the raw result.

Eight captured graphs (six layer/arm graphs and two sequence graphs) retain all tensor owners until replay and exactness checks finish. They are reset before custom allocations are released. Final allocator counters were `live=0`, `destroyed=12`, `cleanup_errors=0`. The launcher sampled the GPU process list 70 times and finished with an empty process list and 4 MiB reported usage. This is sampled ownership evidence, not proof of uninterrupted exclusivity. Elapsed launcher time (85.917 s) includes all checkpoint loading, allocation, exactness, hashing and cleanup; it is not a performance metric.

## What this does and does not establish

Generic compression VMM storage is usable by the original native MoE/TMA path for these fixtures without changing numerical results. The CPU entropy/zlib observation and the actual allocation flag do not establish a GPU compression ratio. This screen supplies no DRAM/L2 counters, so it cannot distinguish poor block compressibility, compression overhead, unchanged effective traffic or other bottlenecks. The data do establish that this allocation-only candidate did not produce the required complete-call improvement in the declared screen. No runtime promotion or further GPU measurement follows from this result.

All original source, protocol, manifest, launcher stdout/stderr and raw checks are preserved. `protocol.json` hash is `fbf8720fc224939a923f3ae2a49f032ca98ff4fb173b29dc1740d65c0fdb877f`; `frozen-inputs.json` hash is `a913a6ae9604ba4648b73491bcf02b4704e8c17f08e6b120bc5a462cdfd2a83c`. CPU preparation passed 108 checks with CUDA uninitialized and Ruff passed. The earlier CPU reports are retained; `cpu-final-source/result.json` matches the executed driver source.
