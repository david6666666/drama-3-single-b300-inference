# Native FP8 storage: bounded CPU feasibility

The designated B300 reports `GENERIC_COMPRESSION_SUPPORTED=1` and VMM support.
This query creates no CUDA context, allocation or GPU kernel. It does not prove
that a later allocation receives compression or that these weights compress.

The frozen sample reads27 original1MiB tensors: gate/up/down for experts0,127,255
at layers0,19,39. Raw SHA256, source headers and original dtype/shape are retained.
Average marginal byte entropy is6.031017bits/byte; CPU zlib
level1 uses0.775649 of original bytes and every roundtrip is exact.
This is a small fixed, unweighted sample. In particular, layer0 contains more zero
bytes than the other sampled layers. Neither entropy nor zlib predicts GPU
hardware compression, random-access decoder throughput or native MoE performance.

The only following hypothesis is transparent VMM compression with unchanged
native FP8 bytes, scales, layouts and arithmetic. Actual allocation properties
must be queried; unsupported or downgraded allocation stops that experiment.
A custom software entropy decoder is not implemented or authorized by this probe.

Official CUDA [compressible-memory documentation](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/virtual-memory-management.html#compressible-memory)
requires the support query and checking actual properties after creation. The
local CUDA header defines the queried attribute as107. Original read-only data,
CPU checks and the no-context assertion are in `probe.py` and `result.json`.
