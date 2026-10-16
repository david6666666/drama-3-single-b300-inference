# Current native pipeline C64 diagnostic

Frozen source e7cda1eba083603b88e1f2e89688e1ff2ced7fc2, GPU0; native pipeline/deferred waiting on, cached Linear off. All1088 requests complete,64 warmup excluded from the event window,105160 output frames retained. The profiled22.8972s/4592.695frames/s is diagnostic-only, not a new quality-qualified performance result or A/B comparison with the earlierGPU6 arm. Clean shutdown and sampled exclusiveGPU0 ownership are recorded.

| Elapsed event accounting | Seconds |
|---|---:|
| Model union |21.2894|
| Codec union |6.6790|
| Model/codec overlap |5.7090|
| Codec only |0.9700|
| Neither model nor codec in envelope |0.6133|
| CPU/client envelope difference |0.0246|

Model stages:1035 ordinary-decode launches15.8038s;196 mixed backbones4.9464s;196 first-frame towers0.2302s;196 old-request mixed towers0.2612s; two standalone prefills0.0478s. Mixed work processes1079 requests/183228 prompt rows in209664 capacity (14.43% row padding), with5.505 new requests and56.061 old requests per mixed call on average.195 spans256 and one512. Current first-frame CPU scope includes5.106s waiting for earlier queued mixed GPU work; it is not5.106s of Python overhead. Pure decode CPU graph submission sums only0.0729s. The uninstrumented complete_decode CPU path is not individually profiled.

The continuous-intake phase lasts21.2193s and delivers100848frames, retaining63.981 client in-flight requests and event-time-weighted60.252 active decode rows. That occupancy distinction matters: client64 is not GPU64. Recorded step counts include680 pairs,310 singles and45 four-step calls over the whole measured interval; executed pure rows94559 include graph work after first-step EOS. The old analyzer's overlap stratification is corrected in a separate artifact to hold both active batch and step count fixed, rather than compare one-frame and two-frame spans as interchangeable.

Within intake, B64 single-step events average9.299ms with codec overlap versus8.562ms without; pair events18.596ms versus16.762ms. All matched B57–64/step groups produce a1.394s association. This is NOT a causal recoverable budget: context, cache, output distribution and which event schedules codec still differ. The approved fixed-input replay tests contention separately. Event unions include possible host launch gaps and are not SM utilization.

The algebraic gap to the historical isolated8.275ms/B64 reference closes to floating-point precision: pure decode above row allowance3.5777s, mixed refill/towers above old-row allowance4.0170s, initial prefill0.0478s, codec-only0.9700s, neither0.6133s, outer0.0246s and executed/productive-row correction0.0500s. These are accounting components, not independent recoverable speedups. In particular, prompt computation is required. The historical7734fps reference remains neither a hardware ceiling nor a guaranteed service floor.

The current graph/schedule does not demonstrate6765fps feasibility. Removing exposed codec-only and neither intervals would save at most the observed1.5833s under an unrealistically unchanged-model counterfactual, far below this run's7.3535s target gap. Native model execution must materially change. The resident-MoE prototype and prefix-state reuse audit are hypotheses for that work; neither has a service speedup yet. Existing user quality and all-concurrency repetition gates stay unchanged.
