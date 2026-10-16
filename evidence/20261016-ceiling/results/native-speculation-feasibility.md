# Exact native two-position verification: CPU feasibility only

Decision: stop the proposed last-frame/history-ngram route before GPU implementation. The two required input codes almost never repeat together in the available real traces. This rejects these cheap proposals as a material optimization, not speculative verification in general. No runtime files, model arithmetic, precision or sampling were changed.

Source inspected:95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b; checkpoint vq_backbone_codebooks=1. `tower.embedding_sum(frames,1,2)` reads frame column2, so the backbone inputs are semantic frame[:,0] and vq1 frame[:,2]. For emitted acoustic frames semantic=semantic_begin+codebook0; DeviceRequests._save stores output columns1..10. Therefore saved/reference code arrays rows0 and1 are exactly the pair to predict. Predicting all10 codebooks would be an unnecessarily strict hit gate, but checking only the semantic code is insufficient.

## Measured causal proposal matches

The CPU driver consumes80 actual generated prefixes from `codec-minlaunch/prefix-replay-arrays/codes/request-*.pt`. Their source8ccefd0 replay reproduced414 archived dynamic trace entries and all80 PCM hashes. They contain6014 emitted frames,5934 transitions; they are partial scripted trajectories, not the new full1088 service outputs. Reference data contains1088 sequences/110478frames/109390transitions;666 unique code arrays remain after exact deduplication,67290frames/66624transitions. Reference audio codes are a separate distribution, not model-sampled acceptance rates. EOS is absent from both acoustic-array sets.

| Proposal | Native generated prefixes | All reference codes | Unique reference codes |
|---|---:|---:|---:|
| Repeat previous semantic+vq1 pair |5/5934=0.0843%|178/109390=0.1627%|119/66624=0.1786%|
| Last observed successor of same pair, fallback previous pair |6/5934=0.1011%|174/109390=0.1591%|113/66624=0.1696%|
| Last observed successor of same pair, no fallback |1/34 proposals;0.0169% of all transitions|18/430;0.0165% of all transitions|11/290;0.0165% of all transitions|
| Oracle selects any previously observed pair |36/5934=0.6067%|437/109390=0.3995%|295/66624=0.4428%|

The oracle row uses the current target to select among old pairs; it is an unimplementable data-dependent ceiling for whole-pair history-copy proposals, not a general bound for learned/factorized proposals. Context lengths2–4 did not improve the generated result. Maps reset per request, update strictly after prediction, and never access future/cross-request targets. Individual generated repeat rates are semantic0.6741% andvq1 2.6626%; their conjunction is the relevant constraint.

Even granting a two-position verifier the cost of one ordinary backbone pass, and zero proposal/state/rollback cost, observed best unconditional p=0.001011 yields at most1+p=1.001011 backbone progress per pass under ideal per-request compaction. This is an optimistic empirical cost illustration, not a hardware lower-bound theorem or service forecast. Actual jointM128 costs, sequential tower work, fallback and heterogeneous acceptance only reduce this benefit. A selective proposal made on34/5934 transitions cannot provide material aggregate coverage either. No GPU timing is warranted for these particular proposals.

## Structural feasibility and exactness obligations

With an exact proposed intermediate frame pair, two backbone token rows per request can in principle share each dense projection/fulltop8 MoE weight pass atM128 instead of twoM64 passes. Each layer would project both rows, execute its causal attention or GDN recurrence in time order, then apply its tokenwise native fulltop8 MoE jointly. The union of experts can be read once instead of two passes; this is the genuine bandwidth opportunity. M128 is not established to cost the same asM64, and BF16/FP8 GEMM shape-dependent arithmetic and routing must pass native exact gates.

That backbone optimization does not vectorize the whole tower. Tower2's first shared layer consumes the actual previous frame's codebooks2..9, which are unknown until tower1 finishes. Native tower1 must sample semantic andvq1 to verify the proposed pair, then finish all residual codebooks. Tower2 can run only after the full verified frame1 exists. Using a teacher-forced tower with guessed labels would not verify the native categorical samples, and reference-code matches cannot justify replacing that work.

For30 GDN layers, sequential decode rounds its recurrent state toBF16 at every token. A two-token kernel must preserve that intermediate boundary, Q/K normalization and exact operation order. Existing chunked prefill is not automatically bit-equivalent. Rejected speculative second tokens require retaining/restoring the state after the valid first token, rather than leaving the rejected second state. ForB64, GDN state alone is30×64×32×128×128×2bytes=1.875GiB; conv histories add90MiB. A simple save/restore design therefore adds significant traffic. Conditional commit could reduce it, but acceptance must already be known across the relevant work and every slot needs the correct first/second version.

Full attention must provide separate causal lengths/positions and distinguish accepted/rejected speculative KV positions. Request history, metadata, output codes, EOS/maxframe masks and every tower KV entry must commit only valid native outputs. Slots cannot be reused before the final state version is settled. Per-row acceptance makes shared-batch scheduling more complex than a single boolean rollback.

The native sampler is deterministic categorical top-p/top-k with seed42 and random coordinates based on(global sampling_step, original row, stream), plus a second semantic sample when the first is repeated in the last16-token history. Grouping128 verification rows naively changes row coordinates. A correct verifier must preserve original per-frame coordinates, repetition history and increments, never resample an already verified frame after rejection, and avoid random-step consumption by discarded hypothetical tower work. Distributional acceptance with new uniform draws would not establish the existing exact RNG contract. The present native tower already increments one global step per batch; per-row acceptance would require explicit contract design before any service implementation.

These obligations do not prove native speculation impossible. They show why the only promising benefit—sharing backbone weight reads—requires a high enough exact proposal rate to pay real additional costs. The measured cheap proposals fail that prerequisite. Any learned proposal/draft model or new precision is outside this audit and outside the currently authorized native optimization scope.

Evidence: [CPU driver](probe_native_speculation_cpu.py), [all counts/file hashes/provenance](native-speculation-cpu.json), [generated-prefix replay](codec-minlaunch/prefix-replay.json). CPU only, CUDA_VISIBLE_DEVICES empty, torch map_location=cpu, no serving/GPU changes.
