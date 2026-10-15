# Reserved-row router footprint, native actual FI routes

Source6ff8d80; GPU6, short padding context1. Actual58/63 initialized prompts, first frame and8 native decode steps. No routing was modified; all80 layer calls captured FI top8 IDs and verified output exactly against calls without capture. No real request had ended.

| Real rows / captured64 | Dummy-exclusive expert-layer pairs | All expert-layer pairs | Logical weight fraction | Dummy-only FP8 weight bytes |
|---|---:|---:|---:|---:|
| 58 | 381 | 6445 | 5.9116% | 1198522368 |
| 63 | 120 | 6326 | 1.8969% | 377487360 |

Logical FP8 weights are3,145,728 bytes per expert (FC1+FC2); this excludes scales and activation traffic and is not a DRAM-counter or latency measurement. The fraction applies only to the sampled partial batches, not all service steps. Existing mixed captures show0,0,3 padding-only experts in layers0,19,39.

The next artifact-only router intervention is assigned to the vllm_scaling agent on GPU2. Source/runtime remains unchanged; real logits/top8/activation quantization must remain exact, only participating reserved rows may change.
