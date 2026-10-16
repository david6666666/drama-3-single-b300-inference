The standalone B300 Drama3 engine still misses the frozen all-concurrency1.5× throughput target. Best complete one-run C64 quality-qualified observation remains4591.038 frames/s versus4510.272 vLLM baseline and6765.407 target; this PR remains draft.

Native full top-8 experts, checkpoint FP8 expert weights/scales, BF16 dense/GDN and complete1088-output WER/SIM gates remain unchanged. Changes are opt-in. Ordered DCO commits retain both initial failures and their corrections.

Latest evidence: [immutable archive](https://github.com/david6666666/drama-3-single-b300-inference/tree/171acc82460f479b56372a57d9f7de7a154795aa/evidence/20261016-ceiling).

- Reserved-padding routing passes2574 exact production checks; actual requests and original quantization remain intact.
- Native MoE residual fusion passes2374 exact checks after preserving the baseline BF16 FMA contraction.
- Native two-frame pipeline passes3689 exact production checks, including slot reuse and a separately labeled synthetic first-EOS case. The callback buffer lifetime fix and validator captured-buffer correction retain their original failed records. Full service/quality is still being evaluated separately.
- Complete codec coalescing pair:4513.299→4414.243 frames/s; both1088-output quality groups pass, throughput regresses.
- Complete48-SM codec pair:4412.554→4470.317 frames/s; WER1.909068%→1.565771%, SIM63.902018→64.244919. Output frames and DP partitions can differ; no fixed-work or repeated speedup claim.

Validation:52 CPU tests for benchmark and pipeline boundaries; frozen dynamic state/PCM gates; complete unfiltered quality for both new codec pairs; archive hashes verified. All-concurrency three-repeat acceptance remains incomplete; no success example or winning configuration is nominated.

Acceptance issue: https://github.com/ThinkFlowLab/OmniRSI/issues/10
