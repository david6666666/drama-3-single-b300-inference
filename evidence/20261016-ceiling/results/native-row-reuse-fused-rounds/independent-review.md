# Independent CPU review

engine_audit found no blocking issue in the three-round driver. AB/BA/AB covers all three cases; every warmup/measured replay restores all111state tensors before the start event. Each pair is followed by fresh same-initial-state control/candidate executions with complete output/state/RNG/mapping/guard checks. Captured graphs, map buffers, snapshots, guarded compact and norm outputs remain retained.

Actual-JIT recording reads existing compiled cache objects and asm/metadata without compile/load/run. The continuation predicate checks everypair and three-round medians. These are same-process/same-capture sequential repetitions, not three independent processes. Launch checks GPU processes before starting; it does not establish continuous exclusivity.
