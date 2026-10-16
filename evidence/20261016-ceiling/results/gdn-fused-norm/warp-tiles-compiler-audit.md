# C3 CPU compilation audit before GPU

Gluon C3 groups four independent native V32 warp subtiles. CPU AOT target is explicit cuda sm103, num_warps4, num_stages3, enable_fp_fusionTrue. `warp-tiles-cpu-compile.json` records hashes and `cuda_initialized=false`. Initial compile attempt failed at a nested SliceLayout axis; correcting only the K slice construction enabled compilation. No GPU was run during that correction.

Candidate TTGIR retains state [4,32,128] layout [1,1,8]/[1,2,16]/[4,1,1] and duplicated Q/K [4,128] layout [1,4]/[1,32]/[4,1]. Existing CPU mapping assertions proved all element mappings match four copies of native V32.

C3 PTX lines489–580: each warp Q/K has local q1*q1 then q0/q2/q3 FMA, XOR16,8,4,2,1 and native div.full/sqrt. The two K dot reductions (lines894–1360 and1572–2080) retain eight local terms ordered1,0,2,3,4,5,6,7 with7FMA, then XOR8,4,2,1. Rank-one state update remains FMA. Operand exchange sometimes reverses multiplicands but preserves finite FP32 products and FMA accumulation order. Beta/raw/state/norm/weighted norm BF16 boundaries remain explicit. Unlike C2, there is no four-warp Q/K reduction.

Norm starts from rounded BF16 raw converted to native one-V-per-thread layout, XOR16,8,4,2,1 then four shared sums XOR2,1. Mean uses explicit div.full, eps add.rn, rsqrt.approx.ftz; norm and gate multiplications are explicitly rounded. Native norm compiler contracts BF16×BF16 with final BF16 rounding to mul.rn.bf16, C3 currently emits FP32 mul.rn plus BF16 conversion: same finite BF16 mathematical product/rounding, but actual exact output gate remains mandatory. No arithmetic equivalence is asserted for nonfinite inputs.

CPU cuobjdump:177 registers/thread,0stack,0local; Triton shared512B (cuobjdump allocation report1024B). Native A168 registers/thread. C3 adds layout-conversion barriers before the same two norm barriers; total emitted bar.sync4. No spill detected, but increased CTA register footprint may cost performance. CPU inspection is enough to attempt the predeclared single real-input exact gate; it is not a numerical PASS or speed claim.
