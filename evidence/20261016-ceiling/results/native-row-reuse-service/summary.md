# Native row reuse: complete service feasibility fails throughput

Frozen source `345c6a18f200853489a9ac8a86bbe0682ebe484d`; one B300 GPU6, C64,
64 warmup and all 1088 measured requests per arm. Only row reuse changes.
All 1808 frozen dependency hashes were independently rechecked after completion.

| Arm | Frames/s | TTFA p50 ms | TTFA p95 ms | Mean RTF | WER % | SIM | Scored |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control, reuse off | 4674.814977 | 148.562946 | 274.704463 | 0.290890 | 1.465293 | 64.188428 | 1088/1088 |
| Candidate, reuse on | 4534.131006 | 151.265127 | 273.353699 | 0.290618 | 1.682994 | 64.202021 | 1088/1088 |

Both complete audio-quality groups pass the confirmed same-C vLLM limits.
Candidate/control throughput is 0.969905981, below the frozen 1.01 continuation
gate. No three-round service experiment follows. This default-off feature is
not a recommended performance configuration. The all-concurrency target remains
INCONCLUSIVE, and the C64 target remains 6765.407263 frames/s.

The candidate activated in 130/200 lifetime opportunities and removed 36096
MoE input rows. CPU plan construction totalled 29.439 ms; this counter excludes
subsequent validation/H2D, which remain included in full service timing.
One-second observed peak process memory increased from 105788 to 106488 MiB.
These sampled peaks are not exact allocator maxima.

Returned frames differ (102266 versus 102637). Intake durations are 20.8292
and 20.7904 s, while last dispatch to final completion is 1.0461 and 1.8441 s.
This client accounting does not establish GPU causality or license deleting
the tail. Full makespan, all outputs and high-WER cases remain included.
The run does not prove a fixed-output kernel regression or a repeated gain.

All four service/quality runners exited successfully and recorded an empty
selected GPU after cleanup. Polling cannot exclude subsecond interference.
The original production correctness harness failure and corrected 9974-check
gate are retained in adjacent evidence directories.

Optimization paused at user request on 2026-10-16. No further GPU work is running
for this experiment. Raw audio stays in the local artifact directory; public
records retain audio hashes and complete timing/quality results.
