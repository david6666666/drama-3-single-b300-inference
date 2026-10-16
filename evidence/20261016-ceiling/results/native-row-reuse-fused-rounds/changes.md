# Three paired rounds, unchanged candidate

Copy the completed fused-adapter helper byte-for-byte and retain its capture/inputs/shape/mapping code. The initial three-case exact gate remains. Timing now runs three rounds AB, BA, AB, each over all three cases with100 independently restored replays/arm and3 warmups/arm in the same order. Save every event duration plus timestamps relative to an arm origin event. Every warmup, timed replay and final correctness replay restores the complete same111-state case snapshot.

After each paired measurement, replay control and candidate anew from that same snapshot, retain independent control outputs/states, and byte-check complete outputs/all111states/both CPU and CUDA RNG/mapping/compactslots/all guards. This is a post-round verification, not a claim to recheck full states after every measured replay.

Require every case in every round to reduce median latency by at least5%, and eachcase three-round median reduction at least5%. Preserve all raw results even if a performance round misses; any numerical failure stops immediately. No runtime/service test follows automatically. Capture/preparation remain separate. Already compiled and launched norm objects are read from Triton device_caches before correctness/timing; writing PTX/CUBIN/metadata does not compile/load/launch a kernel.
