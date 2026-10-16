# Latest-source Drama 3 comparison archive

[The descriptive summary](files/remaining-sglang-report-01/matrix-summary.json) retains two incomplete concurrency rows;
this is not a complete 63-group matrix or an optimization acceptance PASS.
There are 60 complete performance/quality groups (65,280 requests), 19 formal
three-run rows, and 1,087 scored audio outputs from one invalid partial group.
The latter group retains all 1,088 request records including its one HTTP500
failure. Total scored saved audio: 66,367. Partial-output quality is not a score
for the full 1,088-request workload.

vLLM C64 has two complete diagnostic runs; its third run stalled and was stopped
after request-timeout cancellation evidence. SGLang C2 has one complete diagnostic
run and a second invalid run with 1,087 successes and one HTTP500/no-audio response.
Its C2 third run was not attempted. No formal three-run median is reported for
either incomplete row. The invalid group's recorded performance metrics appear
only as failure diagnostics. Duplicated cancellation logs are not unique request counts.

Selected standalone and vLLM records remain under `files/recovery-attempt-01/matrix/`.
The first SGLang server records remain under `files/interrupted-matrix-report-01/sglang/`;
its five groups are scored under `files/remaining-sglang-report-01/existing-quality/`.
Only untouched C4/C8/C16/C32/C64 were measured in the new SGLang server session,
under `files/remaining-sglang-report-01/remaining/`. This new startup/capture/RNG
boundary is explicit; SGLang did not complete all21 groups in one server lifecycle.
The 20 saved vLLM groups are scored under `files/interrupted-matrix-report-01/vllm-quality/`.

All three stops remain preserved: the original monitor timeout, the vLLM service
stall, and the SGLang no-audio response. `files/matrix/` retains the original
monitor-failed attempt; its seven completed groups are excluded from these results.
`files/feasibility/` retains separate pilots, also excluded. No failed row is retried,
replaced or reconstructed. Missing interrupted-group records are not invented.

Raw request JSONL, quality JSON and logs use gzip; small protocols/configurations,
commands, helpers and reviews remain plain. [The index](archive-index.json) records
exact original/uncompressed SHA-256 and byte counts plus stored-file checksums.
Existing gzip bytes are preserved; decompress gzip to recover exact source bytes.

WAV/PCM audio stays on the local experiment machine and is deliberately excluded.
Raw records retain PCM hashes, sample counts and timing provenance; quality records
retain successful scores and the original failed record. Worktrees, Git directories,
model weights, checkpoints and caches are excluded. Local WAV files are required
to rerun scoring; this archive does not contain them.
