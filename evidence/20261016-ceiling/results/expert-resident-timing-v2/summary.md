# Native expert-resident MoE: exact, local performance rejected

The zero-debug resident chain passed all 1,262 exact checks, including live-routing ownership, all 15 full-routing cases, reused graphs, whole-launch `ncta=0`, inactive/padding/tail sentinels, unchanged inputs/weights/RNG and CPU/GPU CUBIN identity. It was slower than the original native prepared MoE in every case. The predeclared continuation gate failed; this candidate is not promoted to the runtime and has no service-performance claim.

The three real physical-B64 fixtures (63 live rows plus one original padding row) regressed by 14.159%, 21.512% and 37.111% for layers 0, 19 and 39. The 12 changed-routing stress cases regressed by 8.595–36.996%; worst-capacity 288-token-tile cases regressed by 22.634–24.534%. These remain separate from real-workload results. Medians below are recomputed over three AB/BA/AB rounds, 100 graph replays per arm per round after five warmups; every raw observation is retained in `timing-gate/result.json`.

| Case | Native µs | Resident µs | Resident latency increase |
|---|---:|---:|---:|
| L0/left_only | 83.274 | 93.841 | 12.689% |
| L0/real | 121.299 | 138.474 | 14.159% |
| L0/right_only | 82.460 | 94.145 | 14.170% |
| L0/rows_reversed | 121.261 | 138.535 | 14.245% |
| L0/worst_global288 | 149.966 | 186.759 | 24.534% |
| L19/left_only | 65.806 | 73.877 | 12.264% |
| L19/real | 100.507 | 122.128 | 21.512% |
| L19/right_only | 70.798 | 76.884 | 8.595% |
| L19/rows_reversed | 100.525 | 122.296 | 21.657% |
| L19/worst_global288 | 150.576 | 184.657 | 22.634% |
| L39/left_only | 64.276 | 76.436 | 18.917% |
| L39/real | 81.407 | 111.619 | 37.111% |
| L39/right_only | 67.167 | 79.861 | 18.899% |
| L39/rows_reversed | 81.270 | 111.337 | 36.996% |
| L39/worst_global288 | 150.359 | 185.240 | 23.198% |

## Boundary and controls

Source: `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`; GPU: B300 `GPU-bc643dc2-a0d1-5ed6-5ecb-0605ca9f340a`. The immutable protocol, full file/input hashes, commands, process audit, logs, storage/affinity controls and result are retained as `timing-protocol.json`, `timing-frozen-inputs.json`, `timing-launch-result.json`, `timing-process-audit.jsonl` and `timing-gpu.log`.

Both arms include live native routing-metadata construction, FC1, native activation and FP8 scale boundaries, FC2, and the original ordered-top8 native finalizer. Both consume the same checkpoint FP8 weights/scales, input FP8 bytes/scales, original top8 IDs and BF16 routing weights. Common input quantization and logits-to-top8 computation are excluded. The native baseline retains tactic `(8,5)` and PDL enabled; the resident implementation has PDL disabled. This is the measured complete-implementation comparison, not a claim that the MMA shape or shared-memory residency alone caused the result. No component profile was collected for this final comparison.

The four-CTA cluster retains native M64/N8/K32 operations and K128 scaling, complete shared activation and two ordered DSM barriers. Removing the four global debug exports still leaves the 2,112-byte shared canary initialization and its publication barrier. Its cost has not been isolated and cannot explain away a 14–37% regression; no canary-only blind sweep is justified.

## Original failures and corrected validation

The original single-chain host tuple mismatch is retained in `../expert-resident-chain`; the corrected launch passed 489 checks in `../expert-resident-chain-v2`. The full debug all256-expert chain passed 1,371 checks in `../expert-resident-full`.

The first zero-debug attempt in `../expert-resident-timing` failed its FC2 checker before timing because it indexed live output with an archived physical routing permutation. Both finalized outputs had already matched. That failure was preserved. The current v2 changes only the checker/protocol: every arm uses its own live map after proving range, bijection, expert ownership, original-token ownership and absolute `mn` prefixes; no output-value matching or relaxed tolerance is used. CPU tests include 15 archived maps, 15 valid within-expert permutations and an invalid-ownership rejection. Actual maps differ from archived maps by 65–434 of 512 entries. Kernel/helper/common bytes and CUBIN are unchanged between the two zero-debug attempts.

## Resources and limits

Actual composite CUBIN SHA256: `7e19a1b5b1fc048682abe3b8ce22e33adc535e3c0ebbcceada43b7f74ae6f1d1`. Launch dynamic shared memory is 78,464 bytes plus 1,024 static bytes = 79,488 bytes per CTA, below the declared 80 KiB ceiling; stack/local/spill checks are zero. The 78,080-byte outer array alone is not the full allocation because pipeline barriers add 384 dynamic bytes.

Register-accounting correction: PTX role hints of 160/48 do not establish a 53,248-register allocation. Actual emitted SASS contains no SETMAXNREG, the header is REG32, and its maximum referenced general register is R29. The earlier one-CTA-per-SM inference is withdrawn; see `../resident-register-accounting-erratum.json` and the independent register audit. Occupancy was not measured.

This is a local prepared-MoE negative result, not a whole-model or all-concurrency benchmark. The user's native-computation constraints, quality gates and all-concurrency 1.5× target remain unchanged and unachieved. The exact components and full-chain oracle remain useful for further native scheduling work; performance continuation requires a new concrete hypothesis and independently frozen experiment.
