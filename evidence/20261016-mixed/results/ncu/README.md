# Native B64 MoE hardware counters

GPU2 counter diagnostic only; native full top-8 FP8 experts, unchanged scales and BF16 dense/GDN. Source 7a8cea15d0df5b822a78e81c2656b39776003da9. No service speedup claim.

| Layer | Projection | Duration us | DRAM read MB | DRAM write MB | Read+write TB/s | DRAM peak % | SM warp-active % | Tensor pipe active % |
|---|---|---|---|---|---|---|---|---|
| 0 | FC1 | 76.320 | 428.168 | 13.847 | 5.792 | 75.54 | 92.25 | 3.88 |
| 0 | FC2 | 41.984 | 214.196 | 3.511 | 5.185 | 67.69 | 87.58 | 3.53 |
| 19 | FC1 | 61.984 | 342.057 | 2.791 | 5.564 | 72.57 | 90.24 | 4.07 |
| 19 | FC2 | 36.352 | 171.153 | 1.774 | 4.757 | 62.10 | 86.20 | 3.46 |
| 39 | FC1 | 48.320 | 243.396 | 2.906 | 5.097 | 66.53 | 88.36 | 4.48 |
| 39 | FC2 | 28.480 | 121.745 | 1.292 | 4.320 | 56.36 | 83.79 | 3.79 |

All six launches required one profiler pass. Real 64-prompt prefill, first frames and eight decode steps ran before cudaProfilerStart. Layers 0, 19 and 39 use saved actual decode activations. All three outputs were finite.

Measured DRAM reads are 1.00052–1.00138 times the logical active-expert weight footprint computed from the same router logits. This suggests little repeated expert-weight traffic in these samples; it does not prove identical tie-breaking between torch.topk and FlashInfer. The low tensor active percentages do not translate to an available 20x speedup: the weight bytes still need to be read.

Cache control is none and clocks are unchanged. Nsight kernel profiling may serialize PDL; these durations must not be compared directly to unprofiled service/graph timing. Three layers at one decode position are insufficient for all-layer or all-concurrency claims.

Commands, executable driver, source hashes, corpus hash, per-request input positions/counts, expert count distributions, raw CSV, profiler log, and .ncu-repz are retained in this directory. GPU2 and both task-owned processes were released after completion.
