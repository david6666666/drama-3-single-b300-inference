# Preserved first exact-gate harness failure

FAIL remains unchanged. All40 layers completed their three-case eager and graph numerical comparisons, but the final sampling_step state check reached a zero-dimensional int64 tensor that the original byte comparator could not view as uint8. This is a checker API failure, not a numerical mismatch, and is not a completed PASS.

The original report, driver and protocol are retained. The new `../native-row-reuse-exact-v2` changes only scalar-safe flattening and CPU comparator boundary tests, then fully repeats capture and validation. Its5393/5393 PASS does not overwrite this failure. See `harness-failure.json` and `summary.json`.

SAFE_TO_ARCHIVE. Raw inputs are retained at this path; repository evidence can reference their immutable hashes rather than duplicating large tensors.
