The runtime now shares backbone work between prefill and decode, isolates asynchronous codec capture workspaces, and provides opt-in native packed-GDN, mixed-layout, empty-convolution and reserved-padding optimizations. Full top-8 experts, checkpoint FP8 weights/scales and BF16 dense/GDN computation are retained.

**The all-concurrency 1.5× target remains unmet; this PR is draft.** The frozen [acceptance issue #10](https://github.com/ThinkFlowLab/OmniRSI/issues/10) requires all 1088 requests at C1/2/4/8/16/32/64, three repetitions, WER at most same-concurrency baseline +1 percentage point and SIM at least baseline −2. Baseline quality covers all 21 groups / 22,848 outputs. No candidate has completed the required passing matrix.

Latest same-source GPU0 C64 feasibility (`a607dff`): control **4353.910** → four-change candidate **4591.038 frames/s**; candidate TTFA p50/p95 **140.774/268.097 ms**, mean RTF **0.295911**, WER **1.925814%**, SIM **63.693694**. Both full 1088-output quality groups pass. This is one pair, with candidate output frames +0.858%; the observed +5.446% is not a repeated or fixed-work speedup claim. It is only 1.0179× the frozen vLLM baseline and below **6765.407 frames/s**.

Validation and retained limitations:

- 8 CPU tests pass. Compact GDN passes 2099 exact checks, layout and empty-convolution changes pass 339 checks each and 339 combined, and explicit reserved-padding contexts pass 1023 checks across nine cases.
- Earlier compact/layout/conv service alone regresses full throughput despite passing quality; deferred codec submission raises frames/s with more generated frames, longer makespan and worse TTFA. Both results remain unpromoted and preserved.
- Dynamic private-capture replay preserves all 414 schedule entries, codes, PCM and states; shared capture reproduces corruption. All new async service configurations retain dedicated capture.
- The full asynchronous interval ledger separates required mixed prompt work, partial decode, codec contention and drain. An isolated ~7734 frames/s B64 result is a workload-specific reference, not a hardware ceiling or service guarantee.
- A restored-state contention probe passes all 60 code/PCM/state checks. Concurrent completion beats serial, but model-high stream priority has no material benefit. Codec partitioning and resource limits remain experimental.
- The initial minimum-launch codec exact gate and MoE residual fusion exact gate failed; original failures are retained. Corrections and numerical diagnosis are separate from service qualification. All experimental switches remain off by default.

See [the analysis](https://github.com/david6666666/drama-3-single-b300-inference/blob/ab2d9015d6b0115530e675f25bbf5f188b1d65b1/docs/high-concurrency-analysis.md) and [indexed full evidence](https://github.com/david6666666/drama-3-single-b300-inference/tree/ab2d9015d6b0115530e675f25bbf5f188b1d65b1/evidence/20261016-ceiling). Raw per-request metrics, complete quality, immutable source/commands and uncertainty are retained. No successful README example is added before all frozen gates pass.
