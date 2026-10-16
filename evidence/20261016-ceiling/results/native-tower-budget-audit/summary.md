# B64 原生 tower 的边界融合预算：CPU 审计

结论：现有证据没有建立一个可实施、尚未试过、可节省 **>0.3 ms/step** 的小型 tower 改动。完整 persistent/cooperative tower 不是被证明不可能，但它需要替换主要 GEMM/attention 实现与跨 CTA 同步，不能用“373 次 launch”直接推导收益。本次停止在 CPU 预算分析，没有新 GPU 运行、运行时修改或性能晋级。

## 证据范围与可复算性

- 完整实际 B64 decode trace：`../../drama3-scale150-20261015/native-bf16-projections-b64/trace.json.gz`（实际路径见 `result.json`），source `e2d5d61c7b9e6cb5524c954605f331b3860e7eca`。GPU6，真实 prompt prefill，BF16 GDN，原生 FP8 MoE，三个 CUDA graph replay。原 CUDA-event 十步计时为 8.535674 ms；profiler 中三个 replay span 平均 8.756004 ms，二者用途不同。
- 新的投影合同：`../native-lt-real-contract/result.json`，source `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`，223 次实际 `model.linear` 调用，保留调用顺序。
- 当前源码检查：runtime `22403f7589a1c9ca6264b003199d078c636196ed`。本报告检查现有结构，但没有声称旧 trace 就是当前生产的约 8.3 ms，更不能据此建立服务 roofline。
- `analyze.py` 仅用 Python 标准库读取旧证据；`result.json` 保存输入 SHA256、三轮原值、分类和单位。没有 CUDA/Torch 导入。按每步 `_load`、最后 backbone `add_norm`、sampling-step increment、主 head 和首 sampler 的连续序列定位边界，并断言每步 13 次 tower attention/FFN、75 次 tower GEMM。

## 实际结构与后端

`src/drama3/model.py:585–640` 为 40 层 backbone；`model.py:642–688` 为 tower layer；`model.py:690–750` 是四个 shared layer 加九个 residual layer，合计 **13 层**。每个 residual layer 的 head/sample 产生下一层的 code 输入，不能直接并行九层投影。semantic head 有 first/retry 两次采样；加九个 residual sample，共 11 次采样。

`src/drama3/kernels.py:274–289` 在 B64 使用 `F.linear`，并没有进入 B2–32 的 FI Cute GEMM。`model.py:327–354` 每层 QKV → 已有 fused QK norm/RoPE/KV write → FI Cute paged attention及reduction → sigmoid、乘 gate → O projection。`model.py:680–688` 已有 packed gate/up 和 SiLU×up。`kernels.py:266–271` 的 12 次 table hidden projection 是直接 `torch.mm(..., out_dtype=float32)`，**不在 223 次 model.linear 记录内**。

223 次中 backbone 为160次，tower为63次；补上12次 direct table GEMM 后 tower共75次。不能用223次总预算当成tower预算，也不能把table投影再提为新方案。

## 历史 trace 的精确成本拆分

每个 replay：backbone+输入925个kernel，span平均7.423898ms；tower+保存输出373个kernel，span平均 **1.332042ms**。tower kernel时间求和1.386224ms，interval union1.317609ms，未被任何tower kernel覆盖的间隙仅 **0.014434ms**。kernel求和含重叠，不能与span相加或当成可消除的wall time。

| tower 项目 | 每步kernel数 | 三次kernel时长和的平均(ms) |
|---|---:|---:|
| BF16 dense / table FP32输出 GEMM | 75 | 0.603024 |
| dense split-K reduction | 27 | 0.089559 |
| paged attention + reduction | 26 | 0.163860 |
| QK norm / RoPE / KV write | 13 | 0.043339 |
| norm / residual norm | 53 | 0.135540 |
| attention sigmoid + multiply | 26 | 0.066102 |
| 11次采样的partition/merge | 22 | 0.077099 |
| FFN SiLU×up | 13 | 0.021791 |
| table gather/where/cast、embedding、控制、保存等 | 118 | 0.185909 |

这些是 kernel 成本归类，不是可直接求和的收益。0.014ms 可见间隙也不是全部潜在融合收益的上界：融合还能减少kernel内部setup、共享数据读写；但它否定了单凭373个边界就认定有0.3ms空泡的依据。

投影按原执行顺序再细分如下；右列是紧随该 GEMM 的 split-K reduction，单列防止误归因。

| tower projection | 次数 | GEMM合计(ms) | reduction合计(ms) |
|---|---:|---:|---:|
| semantic head | 1 | 0.006261 | 0 |
| 首层embedding+hidden projection | 1 | 0.006358 | 0.003435 |
| QKV | 13 | 0.120044 | 0 |
| attention O | 13 | 0.087072 | 0.042955 |
| FFN gate/up | 13 | 0.154274 | 0 |
| FFN down | 13 | 0.106775 | 0.043169 |
| table hidden projection | 12 | 0.064757 | 0 |
| residual codebook head | 9 | 0.057484 | 0 |

## 可实施性与上限

1. **attention gate/epilogue融合**：全tower sigmoid+mul仅0.066ms；QK准备全部0.043ms。融合到原attention/GEMM中需要保持BF16中间舍入和KV写入，收益不会自动等于两个完整kernel时长，单项明显不到0.3ms。
2. **跨FFN阶段常驻**：13层两GEMM、SiLU、前后两个norm连同down的split-K，全部合计仅0.388109ms。省0.3ms相当于去掉77%的整段成本，而两个矩阵乘仍必须执行。原生gate/up+down的逻辑BF16权重共约0.98GB，不能只靠删除边界消灭这些读数与MMA。已有 packed gate/up、SiLU融合仍须保留，不能重复计入。
3. **不split-K或GEMM epilogue变体**：tower reduction全部0.090ms；其消失往往换来主GEMM更慢，不等于净收益。新Lt试验中30次大GDN projection为0.331940→0.331992ms，54次[2048,4096] projection为0.412405→0.412599ms，无被选择的替换。最终223次两臂都仍是F.linear，`replaced_calls=0`，约0.0265ms差是A/A波动；不得重新称为gain或重复generic sweep。
4. **整个tower常驻/cooperative执行**：B64 hidden [64,2048] BF16本身为262144B，已超过已记录设备单block opt-in shared memory 232448B（`../expert-pipeline-cpu-audit.json`），还未放Q/K/V、softmax、FFN中间量。gate/up为1.5MiB，attention output为0.5MiB。可用小row tile分拆，但跨输出channel、全hidden归一化、attention和采样需要额外协作与生命周期设计；不能简单在一个CTA保留全tower。
5. 当前tower每步逻辑dense权重读取：63次intercepted GEMM共1,875,378,176B，加12次table为100,663,296B，合计 **1,976,041,472B**。这是按shape计算的逻辑字节，不是实测DRAM。不同层权重不同、输入经过非线性/注意力/采样，不能跨层共享同一投影结果。实现persistent内核还必须重写现有cuBLAS与FI的设备工作，不能在一个kernel内调用其Python/host launch包装来免费消除边界。
6. **跨帧重叠已做**：`../frame-pipeline-results.md` 已利用原生codebook1依赖解锁后八层tower与下一backbone重叠。B64 14.904320→14.517248ms是两帧合计，节省0.387072ms/对，即约0.193536ms/帧，不是0.387ms/帧。其1792项exact通过也不替代服务质量；不把已集成/已验路线再次提出为本轮新机会。

## 决策与保留的不确定性

本轮不建议再开一个未经预算支持的“persistent tower”GPU sweep。若未来有具体设备设计，最低CPU证据应是完整13层数据依赖、可复用buffer字节/布局、跨CTA同步与sample RNG顺序，且从可消除的实际工作而非kernel个数给出>0.3ms净预算。之后先单真实层输出/KV/untouched/RNG边界，再完整B64与partial slots、原frame/EOS合同，才允许同源计时和完整WER/SIM。新的BF16/FP32运算重排若出现非bitexact，应保留原FAIL并证明数值来源，不能用质量门槛掩盖状态或调度错误。

0.3ms相当于当前约8.3ms纯decode的3.6%，并不大，但旧trace中几个尚可组合的小边界没有自动形成独立、可兑现的0.3ms。当前证据支持的是“尚无可信实现预算”，不是“数学上不能优化tower”，也不是最终所有并发目标已经满足。
