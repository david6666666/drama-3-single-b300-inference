# Transparent compressed VMM allocation: byte and lifetime gate

One GPU4 probe passes all47 assertions on two original1MiB FP8 weight matrices
from layers0 and39. Actual allocation properties report compression0 for the
control and compression1 for the candidate. Tensor dtype/shape/strides and every
byte are unchanged. A captured read-only XOR graph handles changing device masks
0/63/255/0 exactly; it is not a MoE calculation or a compression-ratio benchmark.

A surviving Tensor view keeps the DLPack storage alive after the original Tensor
reference is dropped. After graphs and all owners are removed, five allocations
(including one unconsumed capsule) have been destroyed; custom live count and
cleanup-error count are zero. Device memory returns to4MiB with no compute process.
All cleanup occurs in the retained primary CUDA context and includes a synchronize
before unmapping. Fatal context loss is not a guaranteed cleanup path; a zero
logical live count alone would not prove recovery in that case.

The C++ extension built on CPU with warnings treated as errors. Five negative
CPU cases and import checks pass without CUDA initialization. The one process
probe takes2.467s including CUDA initialization and copies; no performance timing,
cache flush, clock change, native kernel replacement or runtime integration occurs.
The recorded flags establish compressible backing storage, not that these actual
weight contents reduce traffic or improve performance. The next independently
frozen experiment compares the unchanged native MoE using identical weights on
uncompressed and compressed VMM allocations.

See `micro-protocol.json`, `micro-command.json`, `micro-result.json`,
`micro-exit.json`, `build.json`, and original allocator/wrapper source. Retained
owners must outlive every captured graph that references their addresses.
