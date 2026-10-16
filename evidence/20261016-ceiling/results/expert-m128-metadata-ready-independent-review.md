# Independent CPU review: metadata-ready M128 FC2

The final restricted configuration passed independent CPU review. No numerical or GPU result is asserted. The unchanged AOT CUBIN is `ea52321ec64fc9976a306b7c7aeeb0805033c2e024034e5cc6ce266e7d95b170`. This review did not edit the candidate, launch a GPU, or modify runtime source.

## Validator findings and resolution

The initial allowlist omitted configurable native fields: `num_stages_workid=1` and `use_full_tmem_dealloc_barrier=1` were accepted despite changing the fixed schedule. The author preserved that version and replaced the partial allowlist check with canonical checks of every native dataclass field. Raw defaults and their native `compute_warp_layout` values are accepted; the actual assembly recomputes those layout fields. Non-layout settings are fixed.

The native prefetch-depth default is 3 but this FC2 path derives 0. The final factory explicitly sets it to 0 and rejects other values, avoiding any dependency on prior validation before symbolic assembly. Independently checked: raw, computed-layout and repeated validation all pass with prefetch depth 0; changed WorkId depth, full TMEM barrier, per-token scale type, LoadB warp count and prefetch depth all reject. No general native-validator bypass was installed. The validator copies the complete native field set into a separate do-wait=1 configuration, delegates the original checks, then copies back only the native-derived prefetch-depth field. Inspection of the installed validator confirms that is its sole mutation. The installed generic validator continues rejecting the unqualified no-global-wait configuration.

The explicit IO gate bounds `(M,N,K,experts,tokens)` to `(2048,2304,512,256,64)` and the graph token-CTA extent to 288. The eventual driver must invoke `validate_io` before compiling/capturing and retain that IO, compiled function and all buffers. The low-level kernel/config alone does not prove external readiness.

## Why the dependency change can be correct

This retains persistent CLC scheduling and early-exit handling. Without the global WAIT, WorkId and LoadA can read active count and expert/tile metadata before the B/SF task waits. Consequently all metadata producers must finish with memory visibility before FC2 may launch. A PDL launch trigger alone is insufficient.

The intended graph has DEFAULT/full-memory edges from every node captured by the metadata callback to FC2, while retaining the original activation→FC2 PROGRAMMATIC edge. Installed CUDA `cuda.h` documents that zero-initialized edge data serializes complete predecessor execution with downstream memory visibility; source port zero denotes full completion. This supplies routing/count readiness. The retained task-local WAITs supply activation/activation-scale readiness. The AOT PTX has weight TMA at line 353, LoadB WAIT at 857 before activation TMA at 938, and scale-task WAIT at 1671. There is no entry WAIT. These facts support the intended producer separation, not a guaranteed useful overlap or speedup.

## Graph wrapper review and limits

The reviewed `metadata_ready_graph.py` owns capture, records each callback's complete node delta, requires a metadata prefix, verifies the expected body sequence and acyclic dependencies, then adds a direct DEFAULT edge from every prefix node to FC2 without replacing the activation edge. Prefix types include kernel/memset/memcpy, so a multi-node metadata implementation is covered rather than inferred from one kernel name. Raw graph handles and instantiation remain internal to the API. Local cuda-python documentation confirms the capture-info and AddDependencies argument/return ABI used.

The current node snapshot records function, launch geometry, shared memory and topology, but not kernel argument contents or memcpy/memset addresses. It therefore cannot detect every external parameter mutation. Its safety scope must explicitly require trusted callbacks, private graph ownership, no external parameter editing and retention of the exact IO/descriptor/metadata owners. This is not a general protection boundary against arbitrary CUDA graph manipulation. The author was notified to narrow claims or add binding evidence where practical. CPU wrapper tests and the actual driver are separate from this configuration review.

## Required exact gate

Before timing, retain the 15-oracle native-boundary checker, independent live metadata maps, all padding/guard/input/weight/RNG ownership checks, and changed-routing replay through the same sealed graph. The zero-count test must capture real metadata/count producer nodes; a host pre-zero alone does not validate the new graph edge. It must omit activation/finalization and retain valid nonzero launch grids. To prove zero→positive restoration through the same graph, copy counts from retained mutable source scalars (or equivalent); a permanently captured zero_ producer will simply erase a later host restoration. Native activation bytes/scales must be restored for a positive two-GEMM-only replay.

A trusted eager path requires completion of its own metadata event before one launch sequence and forbids metadata mutation between that event and the consumer. It cannot authorize captured replay. Source-level reasoning and a successful AOT do not replace the exact, empty, dependency and lifetime checks on hardware. No performance claim or recommendation for service promotion follows from this review.
