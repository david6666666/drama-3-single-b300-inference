# Native mixed MoE alias observation

**PASS for the declared identity observation. No compaction, alternative arithmetic, performance timing, or runtime change was performed.** Source is the clean frozen `source-codec-cache` worktree at `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`, matching the current service ledger source. Physical GPU6 UUID `GPU-619f7fdc-5f98-8c37-fe89-0465d6130baf` was idle at launch and returned to 4 MiB / 0% after PID 912015 exited.

The original `MixedGraph` executes D64/P1536/span256 with 56 real old requests and eight arriving real prompts. Its original dense, all full-top8 MoE, attention/GDN, teacher tower and reserved-padding behavior remain intact. During graph capture, a wrapper around `model.moe_parts` inserts a copy of each normalized input into a retained, disjoint observer buffer, then calls the original function with the same arguments. The wrapper is removed after capture. The retained 40×1600×2048 BF16 observer is 250 MiB. There is no row gather/compact/expand and no replacement of any native output.

All graph capture/warmup runs precede authentic state initialization. Before each case, states are cleared; 56 original corpus prompts receive native prefill and first-frame sampling. Arriving requests use disjoint private slots, their full original 11-column prompts and position zero. The mixed pass uses the same original graph for all three cases. Its device request buffers and sampling counter remain unchanged, as expected for a teacher-forced mixed backbone pass.

| Case | Proposed duplicate rows | Layers checked | Byte-different pairs | Nonshared controls |
|---|---:|---:|---:|---:|
| FIFO-proxy group190, corpus1040..1047 | 661 | 40 | 0 | 28×40 all different |
| Same group, reversed request order and slots, same graph | 661 | 40 | 0 | 28×40 all different |
| FIFO-proxy group192, corpus1053..1060, same graph | 531 | 40 | 0 | 28×40 all different |

Total **74,120 proposed row-pair/layer comparisons are byte exact**, finite, maximum absolute difference zero. All **3,360 nonshared control row-pair/layer comparisons differ**. Controls were selected at the first different causal input row and excluded by the alias matcher; their numerical inequality is a diagnostic, not an assumption used to allow aliases.

Alias construction checks the real prompt-prefix SHA from the previous CPU audit, full causal prefix bytes, and the exact absolute position. Cross-reference identity is conservatively limited to the previously established common first25 rows; later rows require the same complete reference-prefix identity. Actual graph `frames` and `positions` are checked against the real inputs after replay. Neither matching current text tokens nor reference filenames alone authorize reuse. The three input snapshots retain all actual frames/positions/slots/boundaries, history, request metadata and the explicit alias/control maps.

This validates one previously unknown prerequisite: these actual normalized rows agree in the **original mixed numerical path**. It does **not** validate the next proposed operation. Compressing the GEMM/MoE batch can change native tactic selection, arithmetic and routing metadata order; an independent compact→original-native-MoE→expand exact gate is still required before any timing. Full outputs/state/quality and all-concurrency service targets remain unproven for that future candidate.

The two corpus groups are from the recorded FIFO admission proxy. Although its prompt sums match the old timeline, the service trace lacks request IDs; these are deliberately reconstructed real groups, not proof of actual server group membership. Observation copies alter allocation and launch order, so there is no observer performance claim. There was no uninstrumented-versus-observer full-state A/B: this experiment has an unchanged native compute call and copy-only input observation, not a proposed runtime implementation.

The first CPU import check failed because top-level legacy runtime/Fish imports initialized CUDA. That CPU failure is preserved in `alias-cpu-import-initial.json`; it did not load a model or execute a validation kernel. Before freeze, imports were moved into the GPU-only entry point. The final CPU import/real prompt checks passed with `CUDA_VISIBLE_DEVICES` empty and `torch.cuda.is_initialized()==False`.

Frozen protocol SHA256: `c2e89a7387bc7860775b67f50d038d084247f533749f510614f59da951f451a3`.

Observer driver SHA256: `902a989ac420b2a31596b3685e33683f1362956eee5e3d9bcdf4501c9cdc2087`.

Result SHA256: `cd4e991ffa336c13d8ab450acb19feaf40bc063e424ca28269a9c35d8379975f`.

`alias-protocol.json`, `alias-command.json`, `alias-cpu-check.json`, `alias-result.json`, `alias.log`, driver/CPU checker/launcher and input snapshots form the reproducible evidence. Existing `proposal.md`, `counts.json`, `derive.py`, `provenance.json` remain unchanged. **SAFE_TO_ARCHIVE.**
