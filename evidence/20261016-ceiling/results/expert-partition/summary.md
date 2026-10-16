# Native 2×128 expert partition feasibility

**Exact gate PASS (135 checks); performance gate FAIL; no runtime promotion.** Frozen source `171acc82460f479b56372a57d9f7de7a154795aa`, GPU2 `GPU-bc643dc2-a0d1-5ed6-5ecb-0605ca9f340a`. Process exited normally; GPU2 returned to 4 MiB. Input is archived native FI capture, 63 real rows plus one reserved padding row. No full-model state or sampling was instantiated.

| Layer | Native µs | Serial partition µs | Parallel partition µs | Parallel vs native time | Parallel vs serial time |
|---|---:|---:|---:|---:|---:|
| 0 | 126.874 | 158.976 | 149.197 | +17.59% | -6.15% |
| 19 | 106.906 | 140.390 | 127.590 | +19.35% | -9.12% |
| 39 | 87.552 | 121.958 | 109.157 | +24.68% | -10.50% |

Each row is the median of 15 alternating-order samples, 20 graph replays per sample with five warmup replays. This is one feasibility process, not three independent A/B repetitions or service performance. All three domains used native tile/tactic `[8,5]`; no tactic/split sweep. All exact gates passed before timing began.

The checks cover archived native routing IDs and output; ordinary→prepared reference; exact effective-row FC1 E4M3 bytes and FP32 scale, activation E4M3 bytes and FP32 scale, FC2 BF16; one global top8 FP32 ordered finalizer; exactly one partition owner per route and local range validity; distinct FC2/map buffers; graph replay after row permutations; input/weight hashes and CUDA RNG. Row permutations alter mappings while retaining expert histograms, so new expert-count distributions are not validated by those replay checks. Full state/EOS/cancel tests are unnecessary for promotion because this candidate failed the local performance gate, and remain required for any later candidate that passes.

The measured complete partition path includes input preparation, canonical native routing, int16→int32 IDs copy, each partition's routing metadata/activation/FC1/FC2, stream join, and ordered finalizer. It also includes three explicit q/scale/logits copies to stable prepared pointers; native baseline directly uses input preparation outputs. These copies are visible in `driver.py`, not hidden work. We have no component profile for this experiment, so we cannot assign measured microseconds to the copies, duplicate routing, activation exits, or larger static CTA grids. Parallel being faster than serial partitions establishes a scheduling difference, but is not a kernel-trace proof of a particular FC1/FC2 overlap.

The direct API partition implementation is rejected for service. This does not establish a lower bound for a custom launcher that shares global routing metadata and uses original native GEMM/activation boundaries. CPU follow-up will assess that narrower structural change before any additional GPU work. Actual FP8 intermediate tensors were compared on device, with check shapes/types retained in result.json; no mismatch blobs were produced because every comparison passed. The reference-export API is now exercised, but the transient successful intermediate arrays were not persisted as raw blobs.

Files: `protocol.json` fixes the hypothesis/controls/gates; `driver.py` is immutable experiment code; `launch.py`, `process.json`, `run.log`, `result.json` preserve command, ownership, preparation, raw timings and checks. No source files were changed.
