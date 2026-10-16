# Inverse-aware mixed adapter: three paired rounds passed

**2,430/2,430 byte-exact checks passed.** All nine paired measurements met the unchanged5% threshold, and every case also met it by the three-round median. Each pair has a complete post-timing control/candidate verification of all outputs,111states,CPU/CUDA RNG,mapping,compactslots andguards. No failure occurred.

| Case | AB round1 | BA round2 | AB round3 | Median reduction |
|---|---:|---:|---:|---:|
| proxy190 | 6.25546% | 6.22560% | 6.25623% | 6.25546% |
| proxy190_reversed_reused_graph | 6.24782% | 6.26065% | 6.25987% | 6.25987% |
| proxy192_changed_conditioning_reused_graph | 6.75525% | 6.76730% | 6.75684% | 6.75684% |

Each pair measured100replays/arm after3warmups. Every warmup and measured replay restored the same complete111-state initial snapshot before the start event. Everypair raw start/end relative to its armorigin and elapsed times are in result.json. Restoration is excluded equally; fullmixed nativecompute,gather,inverse-aware norm,teacher andoutputgather are included. These are same-process/same-capture rounds, not independent process runs or serving results. Setup/capture walltimes were separately recorded with synchronization.

Both actual loaded JIT norm variants have32registers,zero spills,staticSHARED1024+dynamic16B. Actual CUBIN hashes differ from offlineAOT because actual launch specializations include pointeralignment metadata; actual-jit.json records signature/constants/attributes andbinary hashes. Candidate arithmetic/helper is unchanged from the preceding singleAB pass.

The graph fixedwork saves roughly1.81–1.98ms in these high-alias D64/P1536 cases. This does not establish service-level throughput gain, coverage of other compact capacities, or6765fps attainment. FIFO proxy membership remains unverified against request-ID logs; full runtime scheduling/quality have not been tested by this artifact.

No runtime source was modified, no new GPU experiment was automatically started, and all earlier failed/accepted artifacts remain intact. GPU6 process exited and returned to4MiB/0% after the run. The launcher audits preflight ownership only; it does not prove continuous exclusion.

Source `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2` remains clean. Protocol `f8ad8e55b9f0f8f3a6c829536e2060ab9f92f6da43a75d0b5845a64b8eb5dd25`. Result `3812a4c6a5cdf59cc12e3e64a215a60e68b6153bbb4335791f22db0e51201118`.
