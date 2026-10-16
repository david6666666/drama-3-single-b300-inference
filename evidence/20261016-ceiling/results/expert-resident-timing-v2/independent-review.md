# Resident zero-debug 完整调用独立复核

结论：**1262 项已记录精确检查通过，但局部续行门槛失败，所有 15 个场景均回退。不得进入整模型或服务晋级。** 本复核只读取、复算已保存证据，没有重新运行 GPU 或改变执行过的脚本/阈值。

## 原始计时复算

| 场景 | native 中位数 ms | resident 中位数 ms | 延迟变化 |
|---|---:|---:|---:|
| L0/left_only | 0.083273916 | 0.093840637 | +12.689% |
| L0/real | 0.121298876 | 0.138473597 | +14.159% |
| L0/right_only | 0.082460480 | 0.094144964 | +14.170% |
| L0/rows_reversed | 0.121261444 | 0.138535357 | +14.245% |
| L0/worst_global288 | 0.149966078 | 0.186759357 | +24.534% |
| L19/left_only | 0.065806398 | 0.073876801 | +12.264% |
| L19/real | 0.100507202 | 0.122128325 | +21.512% |
| L19/right_only | 0.070798402 | 0.076883521 | +8.595% |
| L19/rows_reversed | 0.100525122 | 0.122295675 | +21.657% |
| L19/worst_global288 | 0.150576000 | 0.184656639 | +22.634% |
| L39/left_only | 0.064276161 | 0.076435518 | +18.917% |
| L39/real | 0.081407356 | 0.111618557 | +37.111% |
| L39/right_only | 0.067166719 | 0.079860802 | +18.899% |
| L39/rows_reversed | 0.081270084 | 0.111336956 | +36.996% |
| L39/worst_global288 | 0.150359364 | 0.185239677 | +23.198% |

逐项重新计算 45 条原始记录：每项 3 次 AB/BA/AB，每臂每次 100 次 graph replay、5 次 warmup，全部中位数与原报告相等。三个 real 场景各自至少降低 5% 的门槛都未达到；12 个压力场景也全部超过允许的 5% 回退。机器复核和原始三重复保存在 `independent-review.json`，无跨场景平均掩盖失败。

## 校验器修正与所有权

第一版 `../expert-resident-timing/timing-gate/result.json` 的 exact FAIL 原样保留。它错误地把归档物理排列用于当前 native populate 的输出；当时两臂最终 routed 输出已精确。v2 使用每臂自己的 current expanded→permuted metadata，先验证双射、expert、原 token、absolute mn 的有效前缀，再逐位比较归档 logical expanded 输出。没有按输出数值寻找排列或放宽误差。90 份实际 metadata 快照哈希全部核对；相对归档排列差异为 65–434/512，直接证明不能假定固定物理行顺序。

源码锚点：`timing_driver.py:83` 的 live_metadata；`:121` 的 FC2/zero-debug sentinel 检查；`:152` 的同 body graph ncta0 gate；`:210` 独立存储；`:262` 计时；`:281` 中位数与冻结门槛。eager、复用 graph、empty、四个删除的 global debug 输出 sentinel、inactive FC2、输入/完整权重/RNG、CPU/GPU CUBIN 身份全部有记录。

## 比较边界与资源

输入是相同原生 E4M3 q、FP32 scale、top8 ids 和 BF16 route weights；两臂完整调用均含 live metadata、FC1、原生 activation/FP8 边界、FC2、ordered-top8 native finalizer。两臂共同排除 input quant 和 logits→top8。native PDL=True 保留，resident PDL=False 是候选实现局限，不能称只改 MMA 的单变量实验。MoE 没有递归状态；每场景相同输入、独立 workspace、每次重建 metadata，计时不靠不同输出长度。

实际 CUBIN `7e19a1b5b1fc048682abe3b8ce22e33adc535e3c0ebbcceada43b7f74ae6f1d1`；host launch 的 dynamic shared 78464 + static 1024 = 79488 B/CTA；stack/local/spill=0。**撤回用 160/48 PTX hints 相加成 53248 寄存器来推断驻留 CTA 数的解释**：resident SASS 无 USETMAXREG，实际 header REG32。见 `../register-lowering-audit/result.json` 与 `../expert-resident-timing/register-hint-independent-audit.json`。shared fit、无 spill、source hint 均不是 achieved occupancy 证据。

这只是三个真实 B64（63 live+1 padding）层及 12 个人工路由压力场景；并未测全部 40 层、codec 重叠、当前混合调度或服务吞吐。精度通过不等于性能目标达到，局部失败也不是目标 6765 fps 在硬件上不可能的证明。
