# Native MoE row compaction exact gate

**EXACT PASS: 5,393 / 5,393 checks. No timing or runtime integration.** Frozen source is clean `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`; physical GPU6 is released after PID944978 exited.

The new gate reconstructs the three real mixed cases through the unchanged native model and persists each case's 40 actual normalized MoE input tensors. All120 complete-input SHA256 values match the earlier copy-only identity observation. Each comparison arm receives the same real captured tensor; there is no synthetic or rounded replacement input.

The candidate retains every original decode row and padding row in original order, removes only the explicit equal-causal-prefix aliases, appends copies of retained row0 to a fixed64-aligned capacity1088, calls the original `model.moe_parts(..., decode=True, row_slots=...)`, and expands all three outputs back to all1600 original rows. Logical unique rows are939/939/1069, leaving149/149/19 extra copies. Fixed1088 permits the same per-layer captured graph to process reversed request/slot order and changed reference conditioning. The minimum capacity960 for the first two cases has not been tested.

| Scope | Result |
|---|---|
|40 layers ×3 cases, eager and per-layer graph replay|All passed|
|Actual native preparation FP8 bytes, FP32 scales, router logits, expanded to all1600 rows|Byte exact|
|Native canonical top8 IDs and weights, all original rows|Byte exact|
|Routed expert output, shared expert output and shared gate, all original rows|Byte exact|
|Source/compact inputs, slots and keep/inverse maps|Exact|
|Input/output/map sentinel boundaries|Unchanged|
|Six actual read weight tensors per layer,240 total|Byte unchanged|
|All111 model/request state tensors, including sampling_step|Byte unchanged|
|CPU and CUDA RNG|Byte unchanged|

The native canonical route API independently consumes the actual original preparation outputs. It does not intercept or substitute the opaque internal route buffer of the public MoE call. Both full original MoE calls execute their normal internal router and full top8 computation. There is no tolerance, changed scale, removed expert, or changed precision.

This remains a **component gate on authentic inputs**. The candidate has not yet replaced all40 MoEs inside one complete mixed execution, so the readonly state check is about isolated stateless MoE calls, not an integrated candidate model trajectory. Full mixed output/state/teacher/ownership and fixed-workload graph timing are subsequent gates. There is no service throughput, quality, or6765fps claim. The corpus groups remain a FIFO admission proxy, not measured server request-ID membership.

The first version's original `FAIL` is preserved in `../native-row-reuse-exact/`: its40-layer numerical checks passed, but the final scalar sampling_step comparison raised PyTorch's zero-dimensional dtype-view error. The new version changes only the comparator to flatten before viewing bytes, adds CPU scalar/signed-zero/empty tests, and reruns the full capture+exact protocol. No native arithmetic, alias mapping or numerical criterion changed. The original5,500 successful checks include readonly rechecks in its failure handler and must not be counted as a completed pass.

Protocol SHA256: `babe3f500dab7d3e95d489eb1b09af87d8ac6a1d06a4e74fe13b990b9bbd1042`.

Driver SHA256: `0c90488314c1f2c331fe7a9ade8326735d492c32f7bc3424715176f18e287887`.

Result SHA256: `0657dcd4571c6fa4da37d660aaa5535a1ad316a4219a9f65a6c964aea3441d0b`.

`protocol.json`, `command.json`, `mapping-cpu.json`, `import-cpu.json`, `capture-result.json`, `result.json`, `gate.log`, scripts and the recorded input tensors preserve the experiment. Raw normalized inputs occupy about750MiB per version; JSON records contain their layer hashes. **SAFE_TO_ARCHIVE**, preserving the first failure and linking raw tensors by manifest if repository size limits require separate storage.
