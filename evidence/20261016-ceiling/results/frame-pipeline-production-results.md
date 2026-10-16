# Production two-frame pipeline validation

Frozen DCO runtime: `a7118fb7eb748d08c3b774fa6c9189b6eab0d9cc`. Default-off `DRAMA3_FRAME_PIPELINE` uses the native tower code1 callback; no copied sampling implementation. Reserved-padding routing and five native optimization flags enabled consistently in both arms.

**PASS: 3689 exact checks, 11 cases.** Full B64, partial58, reversed reused slots, max_frames boundary, B32, extra capacities16/24/40/48/56, and a separately labeled synthetic first-EOS case. Frames, full conv/GDN/KV/request/RNG states, nonparticipating ownership, eager/captured hidden2 and states all exact. Source remained clean.

Synthetic EOS executes the native semantic sampler then overrides only row0 stream0 at one declared sampling_step. The second frame remains unforced; all expert/codebook math runs. Sticky done and unchanged row0 request last/history/codes/count/position are verified. This is a boundary diagnostic, not a natural-quality trajectory.

| Real case | Serial two frames ms | Pipeline two frames ms | Reduction ms |
|---|---:|---:|---:|
| real-b64 | 14.824448 | 14.431264 | 0.393184 |
| real-partial58 | 14.828512 | 14.445600 | 0.382913 |
| real-b32 | 12.240896 | 11.831328 | 0.409568 |

Five alternating CUDA-event samples after one warmup, only after all exact cases passed, with full state restoration outside timing. This measures two frames and does not establish a service speedup.

All preceding failures remain visible: (1) a251689 production device ScatterGather assertion; callback-local partial_frame lifetime was shorter than auxiliary use. Retaining it alone passed335 checks and led to the minimal source fix. (2) first fixed driver passed558 checks then compared stale Python hidden2 attribute in the second case. Eager forward rebinds that attribute, while CUDA replay writes the captured storage. V2 pins captured hidden2 and partial-frame buffers, logs pointer differences, and retains the exact hidden2 check. No threshold was relaxed.

Pointer evidence: captured hidden2 address137422077086208 remained fixed; eager rebound self.next_hidden to137421236128256, which was still the Python attribute entering partial58. Captured/eager exact comparison succeeds when it reads the correct fixed capture buffer. Buffers are process-specific addresses, not reusable identifiers.

Artifacts: `frame-pipeline-production-validation.json/log` original failure; `pipeline-lifetime-diagnostic.json/log` isolated335PASS; `frame-pipeline-production-fixed-validation.json/log` stale-handle failure; `frame-pipeline-production-v2-validation.json/log` final3689PASS. All drivers, declarations, hashes and commands are retained. See `pipeline-lifetime-findings.md` and `frame-pipeline-integration-design.md` for causal scope and bounded EOS/admission behavior.

Actual earlier ledger proves scheduled audio boundaries block790/1507 ordinary decode calls with capacities>=16. Remaining717/1507=47.58% is only an upper bound because waiting/pending/full roster were not logged; it is not observed pair coverage. `pair-eligibility-ledger.json` records the reproducible CPU audit. The new service candidate logs actual selection counters at shutdown, including64 warmup and1088 measured requests.
