# Native FlashInfer current mixed diagnostic stopped at its first numerical gate

Frozen source `1618ad4b2c7591465a47ca0dad8100e782820cd2`, same native full-top8/BF16 contract. `DRAMA3_COMPACT_GDN=0`; only existing FLA vs FlashInfer prefill backend differs. The declared same-input RRMS threshold was 0.01 for every sequence, including private padding, with exact ownership and graph replay checks planned separately.

Layer 0 failed: overall output RRMS 0.003518; the six real prompt output sequences were 0.002292–0.003033 and their final states 0.003599–0.004773. The 159-row private padding sequence output was 0.035814, exceeding the threshold; its final-state RRMS was 0.004267. All measured values were finite. Two empty sequences had exact zero state difference.

The driver stopped immediately. No timing samples, later-layer gates, complete trajectory, graph replay checks or service run were performed. This remains a failed gate and is not reclassified as a passing candidate. The location of the failure in private padding does not establish whether either backend is more accurate; an independent FP32 recurrence reference and output-magnitude analysis would be needed before interpreting it. No threshold was relaxed and no retry was performed.

Numerical boundaries: both paths use the existing FLA caller L2 normalization and BF16 Q/K/V; native BF16 pooled state. FLA computes chunk-local cumulative log-decay; FI receives FP32 `exp(decay)` and uses its native SM103 CuTe-DSL non-CP implementation. The explicit `backend="flashinfer"` excludes the separate Cake backend. With 9 sequence slots and 32 state heads, the installed SM100 CP heuristic is false (288 * 4 is not < 148 SMs).

Evidence: `fi-mixed-experiment.json`, `fi-mixed-command.json`, `fi-mixed-validation.json`, `.log`, `validate_fi_mixed.py`. Driver/source hashes and the entire failing row are retained.

After PID 230707 exit: 2, GPU-bc643dc2-a0d1-5ed6-5ecb-0605ca9f340a, 4 MiB, 0 %.
