# Native causal prefix export and continuation: design only

No runtime edits or GPU tests were made for this direction. `build_drama3_prompt` actually places reference text/audio before target text; the previous contrary suspicion was unverified. All1088 requests share25 leading rows, which is shorter than one native64-token GDN chunk.

## Opportunity and readiness

`prefix-reuse-probe.json` reports an empty-cache sequential upper approximation of422 hits /51584 reused rows (27.88% of185006 prompt rows), using every64-row prefix plus one teacher-forcing lookahead row. `prefix-readiness-probe.json` materially narrows this: all matching predecessors are immediately adjacent requests. When cache entries become ready only after a batch, batch6 has63 ready hits /7360 rows (3.98% of prompt rows); batch8 is3.49%. These are synthetic CPU models, not actual C64 admission measurements. The larger sequential number must not be used as the expected C64 benefit.

## State required for a correct cache entry

A key must match all11 frame columns through row n inclusive for reusing rows[0,n): tower labels are shifted by one row. Keep model/source/configuration fixed for cache lifetime. Entries start empty and are published only after their GPU producer completes; never preload corpus-derived data. Request/device/RNG state and outputs are not cached.

For n divisible by64, cache:

- Thirty GDN recurrent checkpoints as FP32 internal accumulators at the beginning of chunk n/64. The installed native FLA `chunk_delta_h.py` already has `TRACK_STATE`, `track_chunk_idx`, and an FP32-only `track_state` hook. Current `gdn.py` passes None/False. The hook stores b_h before processing the selected chunk, preserving native FP32 recurrence between chunks.
- Thirty convolution histories: the final three BF16 packed-input rows before n, each8192 channels. Current final conv pool is the end of the entire prompt and cannot serve as an intermediate checkpoint.
- BF16 K/V rows[0,n) for all10 backbone full-attention and13 teacher-tower layers, with their original absolute RoPE positions.

The current GDN state pool is BF16 and rounds only at the end of prefill. Copying its end-prefix BF16 value would add a rounding absent from uninterrupted prefill. A hit must load the exported FP32 accumulator into private FP32 continuation scratch, run unchanged chunk arithmetic, then cast only its final state into the normal BF16 request pool. Immutable cache entries must never be mutated by an in-place continuation.

Size per n128 entry is about60 MiB GDN +1.41 MiB conv +5.75 MiB KV, or67.16 MiB;16 entries are about1.05 GiB, excluding metadata. The existing FLA hook exports only one checkpoint per sequence, while the CPU probe inserts every64-row key. An implementation must align the cache policy/probe or deliberately support multiple exports; silently assuming all probe hits would be incorrect.

## Concrete continuation changes

1. Add per-sequence absolute base positions and suffix lengths to packed metadata. Keep new request slots and private padding disjoint, and size graph buckets by actual suffix work; retaining original full capacity would waste much of the benefit.
2. Convolution loads t<3 history from an immutable prefix snapshot. Input history and output state must be distinct: the current multi-CTA convolution can otherwise overwrite its final state before an early CTA reads history. Handle suffix lengths0/1/2 explicitly.
3. GDN receives FP32 continuation scratch and still writes the final request state in BF16. Export uses the existing native TRACK_STATE hook at64-aligned boundaries, with no new arithmetic or lossy state compression.
4. Attention must expose cached prefix K/V to suffix queries. Current packed attention only sees the current suffix tensors. A proof path can gather prefix+suffix K/V and use separate Q/KV sequence lengths with tested absolute causal alignment; a paged FI prefill adapter avoids repeated gathers but adds a numerical/backend validation boundary. Do not assume non-square causal mask alignment or exactness without testing it.
5. Teacher-tower suffix rows use original shifted labels and cached tower KV. No cached final hidden vector is required: layer dependencies are fully represented by GDN/conv/KV; suffix embeddings are recomputed normally. Preserve first-frame sampling order and RNG calls.

Exact export/continuation tests should first hold projected q/k/v/decay/beta fixed to prove FP32 state and conv behavior at n64/n128, then test full40-layer hidden/state trajectories and tower KV. Changed GEMM/attention shapes may change numerical results even when native computation is preserved, so complete WER/SIM validation remains necessary. Cache miss, hit, eviction, pending producer, slot reuse, partial suffix and concurrent read ownership require separate checks.

## Within-batch branching is a different architecture

Ready-only reuse misses most adjacent pairs. Running a prefix graph and then a suffix graph introduces another backbone/tower weight pass and may cost more than the saved rows. A more substantial design would deduplicate identical prefix rows inside each layer, share dense/MoE work on unique rows, then branch GDN/conv/KV state before suffix recurrence. This could avoid an extra expert-weight pass but needs explicit layer-level recurrent fan-out and attention prefix sharing; it is not a small cache adapter. No implementation or benefit claim is justified by the current evidence.

Given3.5-4.0% ready prompt-row reuse, even a proportional reduction of the older4.9 s mixed budget would be only about0.17-0.20 s before export/copy/attention overhead, and row count is not proportional to weight-bound MoE time. Retain as a general architectural direction, not a current6765 fps path.
