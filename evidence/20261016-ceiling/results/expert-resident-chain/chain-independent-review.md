# Resident chain 独立 CPU 资源与生命周期审查

审查结论：冻结前的实际四 CTA、单专家组合内核通过 CPU 资源预算检查；未发现阻止第一次 GPU correctness gate 的源码/TMEM 生命周期问题。**尚未运行整链 GPU gate，不能将组件 PASS、AOT PASS 或本审查写成整链精度/性能 PASS。**

审查对象是 `chain-aot-result.json` 所列 CUBIN `9eb4c4f83850b3743118c50628631dfdcf2707fb607306c377ad9a743ceab0b1`，原生源 `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`；没有修改 runtime、共享依赖或已执行的结果。机器资源证据见 `chain-resource-audit/result.json`，源码哈希见 `chain-independent-review.json`。

## 实际资源

从完整内核 host MLIR 的 `CUlaunchConfig.sharedMemBytes` 字段读取 **78,464 B dynamic**；CUBIN 是 **1,024 B static**，合计 **79,488 B/CTA**，比 81,920 B 上限低 2,432 B。不能只引用 outer Array 的 78,080 B：完整六阶段编译另分配了 384 B 管线 barrier 存储，实际 PTX 可见 outer base 后的 78080…78400 地址。`resource_audit.py` 读取真正 launch 字段，未用最大常数或 allocator 估计替代。

完整 CUBIN 的 STACK/LOCAL 都为 0；PTX 无 local load/store，SASS 无 LDL/STL。静态 REG32 不能代表实际寄存器需求。SETMAXNREG 预算是 epilogue 的 8 warps × 32 × 160，加其余 8 warps × 32 × 48，共 **53,248 registers/CTA**。单 CTA 资源合法不意味着四 CTA 在同一 SM、也不是 occupancy/吞吐证明；outer 实际 launch 是 grid4、block512、cluster4。

## TMEM 与阶段交接

`resident_chain_kernel.py` 在 outer 分配一次 32-column TMEM，立即 relinquish 一次，所有阶段借用地址；最后一次消费者完成后仅 dealloc 一次。实际 PTX 恰好各有一个 alloc/relinquish/dealloc 指令。两个 stage body 对 allocator 和 cfg 的最大 TMEM columns 均作不超过32的 trace-time检查。独立地址槽 `[78016,78020)` 不在原生 stage scratch `[0,72704)` 内，不会被阶段初始化覆盖。

每个 stage 调用已安装 Task 的工作和 producer-tail 原语。自写 `ArtifactTask` 仅支持无持久队列、无 Fork、producer stride1 的静态单 tile；未复制已安装 Task 的完整实现，未修改 CUTLASS。native factory 只替换 Task import。`resident-static-task-contract.json` 对 FC1 实际原生配置的端点核实通过；完整 FC2/FC1 AOT 又执行了各自相同拒绝式断言，未放松不支持分支。

所有 stage task 结束后有 TCGEN before-thread-sync fence 与 full CTA barrier，outer 再收敛后才初始化下一阶段。原生 BF16 FC2 epilogue 的实际 PTX 保留 bulk commit、`cp.async.bulk.wait_group.read 0` 和 epilogue barrier，避免输出 TMA 尚在读 scratch 时复用。stage TaskManager 中没有提前 kernel return；valid_rows0 仍执行统一 outer cluster 生命周期。

阶段间 register role 需要按完整内核判断：两段 FC1 和四段 FC2 的 epilogue 始终是 warps0–7、160 registers。FC1 从初始预算 increase160；FC2 的 decrease160 在同一组已经持有160的 warps上执行。其他角色始终48；warp10/11/12改变的是低预算 load 分工。没有发现48预算 warp被当作已经拥有160的 epilogue；该判断仅适用于当前固定配置，不可推广到任意 tactic。

## Shared / DSM 所有权

FC1 `[72704,74816)` 与 local activation `[76928,77984)` 不重叠。全 activation/scales `[72704,76928)` 会覆盖 FC1，但 `resident_dsm_gather` 先全 CTA 收敛并 cluster arrive/wait，保证所有 activation readers 完成且四个rank的local fragment已发布；完成DSM读取后第二组CTA+cluster屏障才允许重用或退出。所有rank包括valid0均参与。full activation持续保活到四个FC2输出块完成。

FC2 shared-B adapter 将普通 STS 改用 AsyncThread producer协议，B写入后带async proxy fence；SF是同步原生FP32 load/store再由release/acquire发布。native UMMA descriptors、M64N8K32/K128累加和原生FP8/scale边界保留。该源码检查不能替代最终FC1 bytes/scales、activation bytes/scales、FC2 BF16逐字节验证。

## 首次 GPU gate 的缺项与范围

`chain_common.build_chain` 已提供 `launch`、五个独立逻辑边界、`immutable_tensors` 与 `owned_tensors`，与 `run_single` ABI匹配。tile/mn/route/ncta/padded纳入immutable；harness在launch前统一设置自己的sentinel，因此wrapper初始fill值差异不改变validator。harness直接检查逻辑地址，不做候选专用排列修正；native finalizer只以候选八条branch替换归档native其余branch，必须称为 **oracle-assisted composition**。

首次执行前仍需 coordinator 冻结全部候选/validator/依赖哈希与协议、确认GPU UUID和进程、设置watchdog、核对GPU实际编译CUBIN与上述CPU身份。gate包括 eager、capture/replay、五边界、partial0…8、非活跃输出/immutable/RNG、原始ordered-top8 finalizer；遇到首差异立即保存并停止，不计时。

这个 first gate 仅L0/expert205单token tile。通过后仍需另外声明并验证全15原生oracle/完整routing、同一graph变输入与metadata，以及多cluster/动态tile生命周期。当前源码尚未提供 full-routing wrapper，不应把准备好的 `run_full` harness等同于实现完成。没有 runtime promotion 或速度结论。
