# Full mixed row reuse: exact pass, continuation fail

**957/957 byte-exact checks passed; every case missed the predeclared5% latency-reduction gate. This candidate is stopped before repeated trials, runtime integration or service benchmarking.**

The source remains clean frozen `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`. GPU6 PID965230 exited and the device returned to4MiB /0%. Only the artifact callback during candidate graph capture changes execution: it gathers valid causal aliases into1088rows, calls the original complete `model.moe_parts` fulltop8, and expands all3 parts to original1600row order. Decode and all original padding remain present. Original attention/GDN, state updates, teacher tower, slot metadata, output gathering and sampling rules are unchanged. All40 layers retain independent guarded buffers, totaling704,027,456 bytes. Peak experiment allocation was159,398,111,744 bytes, including full-state correctness snapshots.

The control and candidate complete `MixedGraph` outputs match for all rows. All111 model/request state tensors match for each of3cases, including nonparticipating slots and sampling_step; CPU/CUDA RNG and all gather/expand/map guards match. The same captured graphs receive forward, reversed request/slot, and changed reference groups. This is a complete mixed graph gate, subsequent to the independent component gate. It is not a full request/service/quality evaluation.

|Case|Control median ms|Candidate median ms|Saved ms|Latency reduction|
|---|---:|---:|---:|---:|
|proxy190|29.130736|27.997727|1.133009|3.8894%|
|proxy190_reversed_reused_graph|29.136928|27.994192|1.142735|3.9219%|
|proxy192_changed_conditioning_reused_graph|29.300735|28.023808|1.276928|4.3580%|

Each case uses one A100→B100 screen and3 warmups per arm. **Every warmup and each of the100 measured replays per arm starts from an identical full111-tensor state restore.** Copies run on the same stream before its start event, so neither arm's measured span includes restoration. Each replay has its own start/end events; event results resolve after botharms finish. The measured span includes actual gather/expand and the whole original mixed backbone plus teacher tower. It excludes standalone preparation, reconstruction, later old-decode/new-first sampling towers and codec. No clock/cache controls were changed. The100 samples are not3independentAB/BA trials, and exact validation precedes timing rather than checking all111states after every timed replay.

All3 reductions are below5%, so `continuation_pass=false` is the declared outcome even though the raw result's execution/exact status is `PASS`. No threshold was changed and no favorable subset chosen. Per-case minimum capacities960/960/1088 were not substituted: this fixed1088 experiment preserves the predeclared same-graph dynamic mapping scope. There was no follow-up bucket sweep.

The experiment establishes that native causal-row reuse can preserve complete mixed numerical behavior on these groups and saves about1.13–1.28ms locally. It does not establish a service improvement or bridge the6765fps target. Current service has differing mixed capacities and concurrent codec/CPU work; multiplying this local reduction by196calls would be only a crude favorable proxy, not a measured benefit. Expert union and first logical expert-weight reads remain unchanged, as the CPU proposal explained. Corpus membership remains a FIFO proxy rather than recorded admission IDs.

Protocol SHA256: `e948b428e10b726bed94c0390be3e386fcbf86c54b323eac4eee7e8e00f59f35`.

Driver SHA256: `210df7fe9ab9d2044dcc7211ed43975135852fe3d7493716b5d1b75d7e27000b`.

Result SHA256: `9c47d787b8a7636a6a755c7b18a2ec5950dcf2b75e5213109244ee44624a331c`.

All raw100-sample arrays, exact checks, code, command, environment, source/data hashes and CPU review contract remain in this directory. **SAFE_TO_ARCHIVE.** Earlier identity/component evidence and original scalar-checker failure remain unchanged in their own directories.
