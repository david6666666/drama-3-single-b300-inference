# FC2 单专家原生合同：初次可行性失败及 CPU 定位

本次状态是 **NATIVE_MATH_FAIL**，并未放宽验收或进入计时。模型及共享环境均未修改。该状态名称来自统一校验器；后续 CPU 证据把本例差异定位为写回布局，不能据状态名称推断 GEMM 算术发生改变。

冻结运行时 `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`，原生 oracle `expert-metadata-rebase/oracle-layer0-real.pt`。使用 layer0 / expert205 的 8 条真实 route，expanded indices 为 `[2,11,30,186,193,378,457,499]`，对应原始 tokens `[0,1,3,23,24,47,57,62]`。本次 FC2 输入始终是原生 activation，不依赖尚待组合的 FC1/activation 候选。

## 输入与资源合同

- activation：E4M3 字节 `[8,512]`，连续逻辑行；FP32 scale `[4,8]`，轴为 K128 block、token。
- 原始 w2：E4M3 `[1,2048,512]`，scale `[1,16,4]`。没有重新量化、转精度或改变 top8。
- 原生 tactic `(8,5)`、MMA `64×8×32`、K128 块累积保持；仅六个 A/B/scale pipeline stage 深度从 8 减为 4，使独立诊断满足共享内存门槛。PDL 在独立算术检查中关闭，不作性能比较。
- CPU AOT 未初始化 CUDA。实际动态共享内存 72,616 字节（host-module.mlir:167,410），CUBIN 静态 2,048，总共 **74,664 < 81,920**。STACK/LOCAL 均 0，没有 PTX local 操作或 SASS spill 指令。
- CPU/GPU CUBIN SHA256 都为 `7bf81f8d8ab8f822c106bedbc8235adb642fe86130340bf6ab18fa424130b1b7`。
- 初次 CPU 准备遇到 safetensors scale 地址未满足 adapter 的 32 字节对齐；原失败保存在 `initial-alignment-failure/`。仅对 w2/s2 做拥有存储的 clone，所有输入文件字节保持相同哈希。
- CUTLASS 4.7.1 仅经进程 sys.path overlay 使用；原 standalone 4.6.2 与 FlashInfer 安装均未修改。

## 原始 GPU 结果和停止行为

GPU7 上单次 eager feasibility：16,384 个 BF16 输出中 14,323 值不同，25,966 字节不同，maxabs 3.4140625，RRMS 1.3210753。其余 **19 项输入、权重、元数据和 CUDA RNG 不变检查均通过**。失败立即停止；没有 graph replay、timing、完整 cluster 或服务运行。原始 `result.json`、`actual-reference.pt`、`protocol.json`、`frozen-inputs.json`、进程及 GPU 前后审计全部保留。

## CPU 独立定位

候选每 16 个 BF16 的顺序为 `[native0,native8,native1,native9,...,native7,native15]`。仅诊断时做：

```python
aligned = candidate.reshape(8,128,8,2).transpose(-1,-2).reshape(8,2048)
```

全部 **32,768 输出字节与 native 完全一致**；8 行各自的排序多重集合也完全一致。另用原生 activation/weight 字节解码到 FP64，独立计算 4 个 K128 dot products 并应用原始 FP32 scale；其最终 BF16 舍入与 native 的 16,384 值全部一致。诊断驱动及数据在 `independent_cpu_layout.py` / `independent-layout-metrics.json`。不能把排列后的诊断 PASS 改写成原始 kernel PASS：消费者要求 kernel 自身写出正确逻辑顺序。

## 源码位置及最小修复设计

安装的 `flashinfer/prims_ts/batched_gemm/gmem_c_resources.py`：

- 2505–2513：DeepSeek 双 epilogue 使用 `base_row_idx = warp*16 + (lane//4)*2`。
- 2521–2560：BF16 TMA 分支把相应 TMEM 的 val0/val1 交给 `_stage_swap_ab_bf16_dsfp8_tma_pair`。
- 1214–1231：该 helper 用 vector2 将 val0/val1 写到相邻 BF16 位置。
- 1111–1118：`_bf16_tma_smem_element_offset` 根据逻辑 row/col 计算 128B swizzle。

这和固定 16 元素排列证据一致：配对值的真实逻辑行是 `logical0 = (base_row//16)*16 + (base_row%16)//2` 和 `logical0+8`。最小 artifact v2 只需在该 helper 按两个逻辑行分别算 swizzle 地址，并以 scalar BF16 store 写回；保留原来的 BF16 conversion、FP32 scaling、K128 累积、TMA commit 与 barriers。bias 为 0，本例无需改 bias 索引；不能由此宣称泛化到其他 bias/act 合同。

下一次 GPU 必须重新冻结修复源码、CPU AOT 产物及实际资源，且仍以原始未重排的 native BF16 全字节 exact 为门槛。当前没有第二次 GPU 运行，未声称性能收益或可生产使用。
