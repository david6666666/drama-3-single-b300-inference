# Shared native routing metadata feasibility

**1373 exact checks PASS; complete-MoE performance FAIL; no runtime promotion.** GPU2 was released to4MiB after normal exit. Frozen framework source95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b; extension SHA43c14f43b679107ea591f4fc73c95faaa84c1f6d4b1e9c5a171fc56079a016b3. Artifact extension removes exactly two finalized-output interface guards and links byte-identical copies of installed native numeric objects. Original source, cached buildfile and copied-input object hashes remain unchanged (`original-files-unchanged.json`).

| Layer | Native µs | Local serial µs | Local parallel µs | Shared serial µs | Shared parallel µs | Shared vs native time |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 125.389 | 148.587 | 141.466 | 146.330 | 139.051 | +10.90% |
| 19 | 105.573 | 129.227 | 121.291 | 126.258 | 118.477 | +12.22% |
| 39 | 86.426 | 109.723 | 103.117 | 107.624 | 100.098 | +15.82% |

Shared metadata improves the otherwise identical direct-stable local-metadata parallel arms by1.71/2.32/2.93%, or2.414/2.814/3.019µs. It remains13.662/12.904/13.672µs slower than the original installed full native call, failing the ≥5% local-net-improvement gate. These are one-process feasibility medians from15 alternating-order samples×20replays, not formal repeated A/B or service claims. Component timings were not profiled; do not assign the remaining loss to a specific kernel from these totals alone.

All arms use original256-expert top8, original E4M3/FP32 boundaries, native GEMM tactic[8,5], native activation and native static176-per-partition grid bound. Stable original input producer writes directly to each retained input binding; the old three extra copies are gone in both local/shared controls. The isolated variable is local routing metadata construction: two installed native populate calls versus one two-CTA rebase kernel consuming one native global router's initialized CTA IDs and non-exiting count. The allocated per-expert-count scratch is never used.

CPU checks:80 actual input-layer matrices plus3 worst-capacity layouts; global288/local176 verified,3 invalid expert-ID/duplicate cases rejected, CUDA uninitialized. GPU checks: three real archived partialB64 layers(63real+1padding), original/reversed/left-only/right-only/worstglobal288 distributions. They cover native-module identity, FP8 input and FC1/activation bytes+scales, effective FC2 BF16, ordered top8 finalize, owner maps/bounds, replay after changing expert histogram, disjoint mutable workspace, restored inputs/weights and unchangedCUDA RNG. No modelstate/sampling lifecycle was instantiated. Every gate passed before timing began.

15 `oracle-layer*-*.pt` files preserve effective per-route native FC1 E4M3/FP32 scales, activation E4M3/FP32 scales, FC2 BF16, native IDs/BF16 weights, input and routed output for the next arithmetic prototype. Empty-partition scenarios are explicitly included. Untouched padding values are not used as arithmetic oracles.

Independent CPU ownership review by mixed_model found no concrete lifetime/fork/join defect invalidating the negative result. The rebase assumes native sorted expert CTA order; inspected native prefix-scan implementation and changed-distribution exact gates support this. An explicit monotonic-order assertion would clarify future protocols, but this completed artifact remains unchanged.

The direct native partition implementation and shared-rebase variant are both retained as negative evidence. Next CPU audit addresses a different producer/consumer structure—expert-cluster FC1→nativeFP8→activation→nativeFP8→FC2—not another split/tile sweep. User throughput/quality targets remain unchanged.
