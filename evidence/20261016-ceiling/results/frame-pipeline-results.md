# Native two-frame backbone/tower overlap

Artifact-only implementation, frozen runtime source `6ff8d8067fb6e96804f49fd4f0c9020fd315e1fc`. GPU6 UUID `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf`. No repository source was changed by this experiment.

## Corrected dependency

Actual checkpoint audio decoder has `vq_backbone_codebooks=1`. `embed_backbone` calls `embedding_sum(frames,1,2)`, which reads frame column2 (codebook1), together with the semantic token at column0. It does not need the remaining codebooks. After four shared tower layers and the first residual layer/sample, the next backbone can therefore overlap the remaining eight residual tower layers without prediction or changed model computation. The earlier assumption that all ten codes must finish before the next backbone was incorrect.

The prototype uses separate attention workspaces, page metadata and positions for p and p+1. Its auxiliary stream waits until code1 is ready; main stream runs the rest of tower1. Main joins auxiliary before tower2 and its sampling_step increment. Backbone state/KV indices<40 and tower KV indices>=40 are disjoint. Both arms use fused MoE residual, compact/layout/conv-guard/short-padding flags consistently.

## Exact gate

**PASS, 1792 exact checks.** Dedicated serial two-frame graph first equals existing `DecodeGraph(steps=2)` for all frames and full model/request/RNG states. Pipeline outputs include full frames and backbone2 hidden; all conv/GDN/KV/request/RNG states, untouched slots and eager-vs-capture are exact. Cases: fullB64, partial58 padded to64, max_frames boundary row without filtering, and fullB32.

The declared reference is native `steps=2`: it advances local frame/history/position unconditionally between model calls while DeviceRequests.save masks appends after EOS/max_frames. This is not a claim of equality to two separate one-step graph replays after EOS. Service admission, early audio delivery and EOS handling must be validated during any integration. The boundary test exercised max_frames; it did not force a first-step EOS token.

## Isolated CUDA-event timing

| Case | Serial two frames ms | Pipeline two frames ms | Reduction ms | Reduction |
|---|---:|---:|---:|---:|
| b64 | 14.904320 | 14.517248 | 0.387072 | 2.597% |
| partial58 | 15.196064 | 14.826592 | 0.369472 | 2.431% |
| b64-maxframes-boundary | 14.922752 | 14.523296 | 0.399456 | 2.677% |
| b32 | 12.212288 | 11.833184 | 0.379104 | 3.104% |

Five alternating samples after one warmup each; identical full-state restoration outside events. Times cover **two** model frames. B64 saves0.387 ms per pair, about0.194 ms per frame when grouped in pairs, not0.387 ms per frame. No service or quality speedup claim is made.

A pair contains one overlap opportunity; a longer continuous pipeline could expose more inter-frame boundaries, but contention, buffering, admissions and output delivery may change the result. This is a design possibility, not a measured extrapolation. The observed saving is smaller than the earlier optimistic tower-tail budget; no hardware attribution is established by this timing alone.

Artifacts: `frame-pipeline-experiment.json`, `frame-pipeline-command.json`, `validate_frame_pipeline.py`, `frame-pipeline-validation.json`, `frame-pipeline-validation.log`.
