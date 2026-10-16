# Native mixed D64/P1280 MoE hardware counters

GPU2 diagnostic only; source 9247e4bf87d6865a53315bc398b4933284d7660c. 58 actual decode requests and 6 arriving prompts, with fixed decode64/prefill1280 rows and span256. Native full top-8 FP8 expert weights/scales and BF16 dense/GDN computation are unchanged.

| Layer | Projection | Duration us | DRAM read MB | DRAM write MB | Read+write TB/s | DRAM peak % | Tensor active % | Read / unique weight bytes |
|---|---|---|---|---|---|---|---|---|
| 0 | FC1 | 125.920 | 543.262 | 44.777 | 4.670 | 60.90 | 21.56 | 1.0119 |
| 0 | FC2 | 74.528 | 272.841 | 41.031 | 4.211 | 54.93 | 18.29 | 1.0164 |
| 19 | FC1 | 120.672 | 504.613 | 42.297 | 4.532 | 59.09 | 22.10 | 1.0110 |
| 19 | FC2 | 74.528 | 253.754 | 38.320 | 3.919 | 51.11 | 17.93 | 1.0168 |
| 39 | FC1 | 119.712 | 485.891 | 42.286 | 4.412 | 57.53 | 22.66 | 1.0118 |
| 39 | FC2 | 74.816 | 244.201 | 38.896 | 3.784 | 49.36 | 18.15 | 1.0170 |

All six launches required one profiler pass. Actual FI routing used 256/238/229 experts in layers0/19/39. The metadata includes the complete per-expert counts, real/padded segment mapping, saved same-input activations and exact FI routing tensors. Capturing routing into a side buffer was bit-exact to ordinary outputs at all three layers; the six profiled kernel calls used the original unmodified arguments. All outputs were finite.

The read counts are close to one copy of each active expert weight, so this sample does not support a large repeated-weight HBM-traffic problem at M1344. The mixed tile signature is 128x32 rather than assuming the previous small-B tile. DRAM throughput is 49.36-60.90% of the metric peak and tensor active is 17.93-22.66%; neither percentage alone identifies recoverable speedup. Activation/output traffic is larger than in B64. No layout changes or autotuning were attempted.

Initialization and all captures ran before actual 64-prompt initialization, first frames and eight decode steps. The last six slots were then reinitialized as new arrivals and the mixed graph ran once. Real prompt rows are1121; prompt padding159 and decode padding6 are retained as in the existing graph. Model loading and initialization are outside cudaProfilerStart/Stop.

Cache control is none; clocks are unchanged. Nsight may serialize PDL and alter cache history. These six isolated kernel durations cannot be substituted for graph or service performance, and three layers/one state do not establish an all-layer/all-concurrency bound. GPU2 was released to4MiB/0% and both task-owned profiler/driver exited.

The prior B64 blocked-layout/autotune experiment already exists in docs/full-corpus-results.md; this diagnostic adds evidence for the actual mixed row distribution and deliberately does not repeat that experiment. Commands, frozen hypothesis/source hashes, report, raw counters and retained formatting error are in this directory.
