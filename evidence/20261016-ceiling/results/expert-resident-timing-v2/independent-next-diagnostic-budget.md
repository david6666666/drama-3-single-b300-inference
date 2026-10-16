# 下一步原生诊断预算（CPU-only）

## 已测事实与权重负载

当前 zero-debug resident 在三个真实层全面变慢，已达到停止晋级条件。原生 MoE 完整调用局部测量是 native metadata+FC1+activation+FC2+ordered-top8 finalizer；并不含共同 input quant/logits router。只有 layer0/19/39 的真实输入，不能称 40 层逐层已测，更不能把三层平均乘40当最新整模型精确成本。

| 层 | distinct experts | token8 groups | native ms | resident ms | 理想每 expert 读一次 / 每 group 读一次逻辑权重 MB | native 逻辑载荷/调用时间 TB/s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 206 | 208 | 0.121299 | 0.138474 | 648.178 / 654.471 | 5.344 / 5.396 |
| 19 | 159 | 168 | 0.100507 | 0.122128 | 500.293 / 528.611 | 4.978 / 5.259 |
| 39 | 115 | 146 | 0.081407 | 0.111619 | 361.847 / 459.388 | 4.445 / 5.643 |

每专家原生权重 3,146,496 B：w1 E4M3 1024×2048、w2 E4M3 2048×512、原 FP32 scales 768 B。两种载荷只是可复算的逻辑读量情景；真实 HBM 可能因 L2 命中更低、重复读/事务更高。**没有测 DRAM bytes，不能叫 achieved HBM bandwidth 或 roofline。** Layer39 routes 更集中（115 experts/146 groups），native 更快，而固定 phase resident 的差距反而更大；这与固定调度/同步代价占比增大的解释相容，但不唯一证明它。

相比数百 MB 权重，512 个有效 expanded row 的 FC1 global 输出约 0.5 MiB+16 KiB scales，activation 0.25 MiB+8 KiB scales。即使消去其写入和后续读取，也不能假设这些全都来自 HBM：native consumer 可以从缓存读，且读取次数与 output tiles 有关。resident 保留最终 FC2 2 MiB 输出和 native finalizer。直接按“少两次 global buffer”预言大收益没有可靠预算。

## 两种调度的可证差异

`resident_chain_host.py:111` 固定 grid[4,288,1]/block512/cluster4；`resident_chain_kernel.py:97` 每 rank 两次 FC1 phase（up/gate），`:154` 四次 FC2 phase。每 phase native producer drain、TCGEN fence 与 CTA publication，activation/DSM gather 另有 cluster publish/reader completion。native GEMM 把独立 output tiles 并行调度；resident 将其中工作串接进四个长 CTA，并引入 cluster 共存约束。四阶段 staging 与 vendor 八阶段也不同。实际总 shared79488 B/CTA、zero spill 有证据，但 PTX160/48 hint 没有在候选 SASS lowering 成 USETMAXREG，不能继续用53248寄存器解释驻留数。

另一个实质差异是 native PDL=True、resident=False。当前测量是完整实现对照，不能从它单独分离 cluster、stage 深度、barrier、PDL 各项损失。这些是应被证伪的候选解释，不是已测硬件 stall 原因。

## 只允许能回答明确问题的下一步

1. 优先复用现有输出与 CUBIN 做源码/资源归因（已完成），禁止把 maxnreg cap 或大范围 block/tile/stage 搜索当原因分析。
2. 若批准一次诊断 profile，只取 real 的 layer0 与 layer39、两实现相同 q/ids/weight 边界；收集实际 kernel 时长、DRAM/L2 bytes、tensor-core issue、barrier/wait、active blocks/SM/cluster launch 信息。先解决“权重带宽、phase 尾部、流水深度还是 launch 依赖”的歧义；不重跑服务、整模型或全部压力场景做盲测。NCU replay 若改变 graph/PDL，必须标明不与已有 CUDA-event 时间混同。
3. Parent 已另行批准 M128 PDL 单变量：native 基线仍 True，候选从 False→True；它直接排查可操作的依赖开销，不能冒称 resident 已恢复收益。先所有 logical/native 边界、empty 生命周期及 graph node 属性证明，后最短一轮 real feasibility；失败即停。

当前服务 ledger 只有阶段 spans，不能给出最新每一层 MoE 的 HBM roofline。最早 B64 profiler 的 FC1/FC2 总和、早期层0/19/39的139/127/106us也不能直接覆盖本轮121/101/81us（输入、padding与测量边界不同）。现有证据足以否决这版 resident，尚不足以宣称6765fps不可达或已发现可恢复多毫秒预算。
