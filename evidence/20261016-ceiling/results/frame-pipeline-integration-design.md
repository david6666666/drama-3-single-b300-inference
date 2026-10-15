# Bounded native two-frame pipeline integration design

CPU-only source review; runtime source not edited. Evidence: frame-pipeline-validation.json (1792 exact checks), frozen 6ff8d8067fb6e96804f49fd4f0c9020fd315e1fc. Service review: engine.py at 69d636ef11875add6e4edf7c9bb028b9574143a3. Concurrent router changes belong to the vLLM agent and pass row_slots explicitly.

## Measured scope

Use a separate default-off two-frame decode graph after router freeze. Preserve one-frame, first-frame, mixed prefill, and low-concurrency 2/4-step paths. Match existing batch capacities: rounding B40 to B64 merely to reuse a pipeline could erase the gain. Initially target existing capacities >=16; additional capacities need smoke/ownership checks beyond validated B32/B64/partial58.

B64 saves 0.387072 ms per TWO frames (14.904320 to 14.517248), about 0.194 ms/frame in pairs. B32 and partial58 also pass. This is a modest exact structural optimization, not 0.8 ms/frame or an end-to-end gain. Longer continuous pipelines are separate experiments.

## Native tower callback seam

Append optional keyword-only code1_callback=None to DramaModel.tower_forward. Invoke it only with labels=None immediately after residual layer index0 appends sampled codebook1; pass (main, codes[0], codes[1]). Retain the existing loop, semantic/retry sampling, all nine residual heads, return value and sampling_step increment. Reject callback use with teacher forcing. The new graph constructor asserts vq_backbone_codebooks==1, four shared layers and nine residual layers, matching the dependency proven for this checkpoint.

This avoids maintaining duplicate tower_prefix/tower_tail implementations. Existing callers omit the callback and retain behavior. The callback creates partial frame (semantic/code0/code1, other columns zero), forks the auxiliary stream and launches backbone2 at p+1. A local closure/list holds hidden2 during capture; no mutable model slot/hidden context is added. The router's explicit row_slots is naturally propagated through backbone_forward.

The graph owns dedicated capture/auxiliary streams, p/p+1 tensors and TWO AttentionBatch objects with distinct page tables, lengths and FlashInfer workspaces. It loads inputs once, runs backbone1, prepares p+1 with existing max_length clamp, then calls native tower_forward with the callback. Main runs tower1 tail while auxiliary runs backbone2. Main saves frame1, shifts local history, joins auxiliary, then calls ordinary tower_forward for frame2 and saves it. Return [2,B,11], with steps=2/size/run(active) compatible with complete_decode and timeline metadata. No roster change or slot reuse inside the pair.

Backbone state/KV indices<40 and tower KV>=40 are disjoint. Join before tower2 increments sampling_step or reads full frame1 codes2..9. Dedicated capture streams isolate ATen/cuBLAS workspaces from asynchronous codec; do not casually share graph memory pools across concurrent streams. Artifact validation had moe_shared_stream=None; reject simultaneous shared-expert overlap until explicitly validated.

## Bounded admission/audio policy

At ordinary decode selection, after cancellation cleanup and admission/prefill decisions, select a pair only if:

- flag enabled and the current captured capacity has a pipeline graph;
- waiting is empty and pending.empty() at selection;
- every active request has min(next_emit,max_frames)-frame_count >= 2;
- this is not a mixed backbone, first-frame or prefill operation.

Keep row order and call unchanged complete_decode once. There is no scheduled audio boundary inside the pair: if next_emit/max_frames would occur after one frame, use one step. Arrivals after the queue check and cancellations can wait one extra decode frame relative to single-step scheduling. This whole-batch horizon restriction may make many C64 iterations ineligible; record selected pairs and fallback reasons before estimating service benefit. Splitting the roster adds expert weight passes and changes row-based sampler assignments, so is excluded.

Keep deferred codec flush immediately after graph.run enqueues, before complete_decode synchronizes output tokens. Do not add a completion synchronize before flushing. Codec overlap under the additional auxiliary stream still needs service measurement.

## EOS: no extra output, but explicit extra computation

runtime_state._save uses append = !done && main!=EOS && count<limit, with sticky done. A second frame after EOS/max_frames cannot update last/history/codes or increment request metadata. complete_decode examines each request's main tokens chronologically, increments only non-EOS counts/positions, and breaks at FIRST EOS/max_frames. It emits/finishes/frees once and ignores later pair outputs. Audio gathering uses only saved valid code positions.

A full second native forward DOES still execute for a row whose first output is EOS. This matches the validated DecodeGraph(steps=2), not two independently scheduled one-step calls after EOS. Finished row model KV/state can advance again; no slot reuse occurs until CPU completion, and subsequent prefill reset/length ownership handles reuse. sampling_step increments twice globally; row positions stay fixed inside the pair. Do not hide this work or claim no EOS delay: EOS detection, final audio and slot reuse wait until pair completion, at most one additional frame of model work. Predicting/skipping EOS is excluded.

The existing gate covered max_frames but did NOT force first-step EOS. Add first-EOS/second-EOS/normal pair completion tests, request append masking, surviving rows/reused slots and exact capture/eager/native-steps2 comparison. A natural EOS state from a real trajectory is preferable for the full GPU gate; synthetic pairs can independently test CPU completion without changing any performance run.

## Next verification

Freeze combined source, then check callback default path exact, pipeline vs serial native steps2 on full/partial B64 and B32 plus new capacities, full model/request/RNG state, all active/nonparticipating ownership, eager/capture, max_frames and EOS; repeat with explicit router row_slots. Then one full1088 feasibility with identical controls, pair/fallback counts, throughput/TTFA/RTF and full quality. Cover audio boundaries, slot reuse, cancellation and arrivals during pairs. User gate remains native full top8/original precisions, WER <= baseline+1 percentage point and SIM >= baseline-2. No current result reaches or proves inability to reach 6765.407263 fps.
