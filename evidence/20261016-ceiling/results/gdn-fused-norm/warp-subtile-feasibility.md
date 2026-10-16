# GDN 四个原生 V32 warp + gated norm：CPU 可行性审计

结论：**布局上可表达，值得一次受限的编译验证；当前不能声称精确或更快。** 推荐 Gluon 显式三维布局，把四个原生 V32 / 单 warp CTA 合并为同一 CTA 的四个 warp。每个 warp 独立完成原来的 Q/K 归约和递归计算，只有 128 维 gated norm 需要跨 warp 汇总。不要继续把 BV 改成 128 后让编译器自动选择四 warp 归约布局。

本报告只读冻结 source `463d837cd51fb62196378835a33731cb659a66c8`、已有失败产物、已安装 Triton 3.7.1 源码及编译缓存；只运行 CPU 布局映射断言和现有 cubin 的 `cuobjdump --dump-resource-usage`。**没有 GPU 执行、候选编译、runtime 修改或门槛变更。** 原 `first-layer-result.json` 的 EXACT_FAIL 和 `expanded-result.json` 的 NUMERIC_FAIL 保留；没有延迟测量。

## 为什么 C2 仍改变浮点结果

原生 A 的缓存 [A-native-cached.ttgir](A-native-cached.ttgir:1) 中，Q/K 使用 `sizePerThread=[4], threadsPerWarp=[32], warpsPerCTA=[1]`：每 lane 四个连续 K 元素，每个 V32 CTA 各自归约完整 K128。之前 B/C2 的 V128 / 四 warp 方案把 Q/K 分到四个 warp、每 lane 一个元素，改变了求和树；模型方程未改，浮点运算顺序改了。

A 的 [PTX](A-native-cached.ptx:526) 给出具体 Q/K 局部顺序：先 `q1*q1`，再以 q0、q2、q3 顺序做 FMA，然后 XOR 16、8、4、2、1 的 warp 加法。A 的矩阵布局为 `Blocked([1,8],[2,16],[1,1],[1,0])`。每个 K 归约有八个本地乘积：先第 1 项的乘法，再以第 0、2、3、4、5、6、7 项做 FMA，最后 XOR 8、4、2、1；`h*k` 和 `h*q` 都需保留这个序列（PTX:894、915、1029）。原 C 关闭全局 FP fusion，连这些局部 FMA 链也拆开了。C2 恢复了 native B 的 recurrence，却没有恢复 A 的 Q/K 归约布局。

另一个差异是 norm：C2 沿用矩阵的 slice 布局进行 128V 求和，而独立 native norm 是 128 线程各持一个 V。C2 在新真实输入上 state/raw 与 B 完全相同，但仍有 raw/norm 的 A-drift 超界，并有一个 norm head 超过独立 FP64 参考的 1% 退化界限，见 [expanded-failure-localization.json](expanded-failure-localization.json)。因此不能用“小绝对误差”撤销失败，也不能只修 rank-one FMA 就宣称保留了原计算顺序。

## 建议的精确布局

以下坐标中的 `w` 是 V 子块编号，`v` 是子块内 V，实际 V=`32*w+v`；每个 CTA 仍只服务同一个 request/head。

| 阶段 | 逻辑 shape | Gluon BlockedLayout（size_per_thread / threads_per_warp / warps_per_cta / order） |
| --- | --- | --- |
| state 及 recurrence | `[4,32,128]` | `[1,1,8] / [1,2,16] / [4,1,1] / [2,1,0]` |
| Q/K 独立副本 | `[4,128]` | `[1,4] / [1,32] / [4,1] / [1,0]` |
| warp 内 raw 输出重排 | `[4,32]` | `[1,1] / [1,32] / [4,1] / [1,0]` |
| norm | flatten 为 `[128]` | `[1] / [32] / [4] / [0]` |

已用安装版 `BlockedLayout.format_hardware_view` 在 CPU 上逐元素核对：四个 warp 的 **16,384 个 state 元素**都等于 A 的一个 V32 warp 映射加上不同 `w`；**512 个 Q/K 元素**均为 `K=4*lane+register`，register 为 0..3；norm 的 **128 个元素**均为 `V=32*warp+lane`。这些断言只证明映射，不证明生成的数学指令。

recurrence 完成后先把 raw 转成 BF16，随后 `convert_layout` 从 state 的 K-reduced slice 转为表中 `[4,32]` 输出布局，再 flatten。这个转换在每个 warp 内重排自身 32 个 V；**不需要把整个 state 或完整 raw 向量跨 warp 搬运**。跨 warp 只交换四个 norm partial sum。state 写回仍是 BF16，slot 索引、负 slot 行处理、padding 行语义保持原样；不得在需要 CTA barrier 的区域只让部分 warp return。

Gluon 当前已提供 [BlockedLayout](/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/triton/experimental/gluon/language/_layouts.py:60)、[SliceLayout](/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/triton/experimental/gluon/language/_layouts.py:122)、[DistributedLinearLayout](/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/triton/experimental/gluon/language/_layouts.py:165)、[convert_layout](/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/triton/experimental/gluon/language/_core.py:428) 和 [allocate_shared_memory](/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/triton/experimental/gluon/language/_core.py:487)。上述 BlockedLayout 已足够表达元素映射，暂不需要手写 DistributedLinearLayout 基向量或整段 CUDA。

## norm 的确切顺序和同步

匹配冻结源码 `kernels.py:185`、EPS=1e-6、BF16、4 warp、`enable_fp_fusion=False` 的缓存：

`/home/zjy/.cache/sglang/triton/5EVRQCWIS2HNEAFKWCVYXZJOUK6UIGUQ7NDJAVJROSTDVJLOP7OA/gated_norm_kernel.{ptx,ttgir,json}`

PTX SHA256=`35ed1b3c14ff467e8bb64027cba343633219edbb1efd6c90632efc8f5fb7ba25`；TTGIR SHA256=`df8ffa20fb2f080a0710c5a35a1d9fa4fb08ad083c4c1329292536402baf96a4`。

1. 每个 lane 加载对应 raw BF16、扩为 FP32，再做独立 `mul.rn.f32(x,x)`。
2. 每 warp XOR 16→8→4→2→1，以独立 FP32 add 归约。各 warp lane0 写自己的 FP32 partial 到 shared 的四个地址。
3. CTA barrier；warp0 的 lane0..3 读四个 partial，XOR 2→1，得到 `((p0+p2)+(p1+p3))`；lane0 写 shared 标量。
4. CTA barrier；所有线程读取标量，`div.full.f32(sum,128)`、独立 `add.rn.f32(eps)`、`rsqrt.approx.ftz.f32`。
5. 乘 x 后 BF16 round；乘 weight 后再次 BF16 round；按原 sigmoid、`z*sigmoid(z)`、乘 scaled、最终 BF16 store 顺序完成。z 可直接按原投影的 stride12352 读取，免去 `.contiguous()`。

必须保留 beta/raw/norm/weighted-norm/state/output 的原 BF16 边界；尤其 norm 使用 **已 BF16 舍入的 raw**，而非 recurrence FP32 累加器。启用 recurrence 的原 FP fusion 时，norm 的平方、均值和 eps 加法需要显式不融合操作；为严格复现已缓存 PTX，不应擅自把 `div.full` 换成乘倒数或换 exp/sigmoid helper。

最小实现先用 Gluon 固定布局，让编译器产生归约；随后在 CPU 编译产物中逐段检查 Q/K 的 FMA 链、两个 K dot 的 FMA/warp shuffle、norm 的两级树及所有 BF16 convert。如果布局正确但局部指令顺序不同，再仅对这些小归约使用 inline PTX 的 `fma.rn`/`add.rn`/`shfl.sync.bfly` 固定顺序。Gluon 的 `inline_asm_elementwise` 适合已有显式布局内的 lane 运算；**不要在 elementwise asm 中隐藏跨 CTA/warp 的 shared 生命周期或 barrier**，这些由显式布局转换/共享内存原语管理。完整重写 CUDA recurrence 会增加维护面，目前收益预算不足以支持这条先行路线。

## 资源和收益预算

原 A cubin 的 CPU resource dump 为 **168 registers/thread、STACK=0、LOCAL=0**；norm 为 **16 registers/thread**。A 的 Triton metadata 请求 shared64B，norm 请求16B；`cuobjdump` 对二者均报告 SHARED1024B，二者口径不同，不能把 requested bytes 当硬件最终分配。

- state 每 lane 的 FP32 活跃数据下限为 `32*128/32=128` registers；四 warp CTA 本身不会把每 lane 的 state 翻四倍。按 A 的实际168寄存器估算，每 CTA 约 **21,504 个32位寄存器**，原一个 warp CTA 为5,376。融合后真实峰值、分配取整与 spill 尚未知。
- warp 内 V 重排的逻辑 scratch 可按每 warp32个BF16估计，共256B；四个 norm partial 仅16B。若用 FP32暂存 raw，向量本身为512B。只在阶段生命周期不重叠且有足够同步时复用空间；实际 shared allocation 必须看新编译产物，不承诺它与上述逻辑字节数相同。
- B64 每层 raw 为 `64*32*128*2=524,288B`。去掉 raw 全局 store+reload 省1MiB，去掉 z contiguous copy 的 read+write再省1MiB。30层合计 **60MiB/step 的逻辑全局访问**；最终 output store、z 最终读取和 state读写都仍保留。
- 同时原生 state 的每层读+写为 `64*32*128*128*2*2=128MiB`，30层3.75GiB。上述60MiB只相当于这部分的 **1.5625%**，且可能命中 cache，不能按字节比例推导端到端加速。
- 每层 recurrence + z-copy + norm 的3个 GPU kernel可变为1个，30层少60个 GPU kernel/graph node。**它们已处于 CUDA graph 内，不等于少60次 Python/host launch。** 新 kernel 仍需布局转换和至少两个 norm CTA barrier。
- B64 的 recurrence CTA 从8192个单warp改为2048个四warp，总 recurrence warp 数相同。每 CTA 寄存器和同步粒度增大，可能使调度/occupancy下降；不能因 launch 数下降就假设速度上升。此前 fullhead 候选的171/179 registers也不能当作该布局的实际资源数。

这是压缩 recurrence 后处理与 kernel 边界的小收益路线，不减少 state主流量、conv、MoE 或 model compute；目前证据不能支持大幅吞吐提升。此前 rough 的 conv/norm/copy 预算也不能全部计作可省部分，因为本方案未融合 conv。

## 最小继续条件

1. 先只做 artifact 内的 **B64/B>32 BF16** 显式布局版本和 CPU 编译/PTX检查；B≤32原生用BV16，应保留旧路径，不能用本次V32布局证明它的精确性。若需扩展BV16，必须另审计原布局，不能静默替换。
2. 若编译后关键 reduction/FMA/rounding 树未保住、出现明显spill或资源膨胀，停止这条融合；不要立刻扩大实现或回到宽松阈值。
3. 只有父任务另行批准 GPU 后，先重放 **此前 expanded 的失败真实 fixture**，要求 state/raw/norm 与原 A **bitwise exact**、所有不参与 slot exact；这是该“保留原顺序”设计必须兑现的主张。首差异存档并停止晋级。
4. 通过首例才扩展30层、B64/partial58与槽位状态，旧路径覆盖B1/8/32，随后全图codes/EOS与native状态检查。原数值/FP64门槛不改，原C/C2失败记录不变。
5. 正确性全部通过后才测 combined recurrence+zcopy+norm；**至少5%局部收益**门槛不变。端到端同源1088及全量WER/SIM仍是服务晋级必要条件，局部PASS不能替代。
