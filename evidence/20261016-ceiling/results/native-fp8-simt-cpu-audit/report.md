# 固定 8-lane grouped FP8 SIMT：CPU/AOT 后停止

结论：**这一固定实现不进入 GPU 测试**。sm_103a 编译资源可容纳，所选真实 FC2 输出在 CPU 模拟中也与原生 BF16 相等，但实际 SASS 的 FP8 解码与标量算术指令预算不足以支持替换当前约 4 ms/step 的四十层 MoE 两次 GEMM。这里否定的是本固定映射，不是证明所有 SIMT 设计或最终 6765 frames/s 目标不可能。

## 实现与原生合同

`fixed_subwarp8.cu` 固定每输出 8 lanes、每 CTA 128 threads/16 outputs，每个专家任务最多两条原始 route；同一权重字节在两条 route 间复用。每 lane 在 K128 内依次处理 4 chunks × 4 values，以 FP32 FMA 累加，随后按 shuffle 4→2→1 归约。E4M3→FP16→FP32 仅为精确值解码，FP8 有限值均能被 FP16 精确表示，没有在 FP16 中做反量化或点积。每 K128 仅执行一次原 FP32 activation-scale × weight-scale 的 RN 乘法，再按原 K 顺序执行 FP32 FMA.FTZ。FC2 最后一次 BF16 RN 转换保留。

本文件的 FC1 仅导出前量化 raw FP32，**没有实现原 FC1 FP8+FP32 scale epilogue、native activation 或 ordered-top8 finalizer**，不能作为完整 MoE 运行。没有 launch wrapper、生产接入或 GPU 执行。`rows` 必须由未来 host 限定为 0/1/2；当前两个固定输出宽度 1024/2048 整除 16，保证整个 warp 无不均匀尾部退出，full-mask shuffle 合法。其他 shape/rows 不受支持。

原生尺度合同的本地证据为 installed `smem_deepseek_sf_resources.py` 的 scale multiply/`accumulate_scaled_t2r`，以及 `expert-fc2-native-v2/candidate-fc2.sass`：非 FTZ FMUL 后为 FFMA2.FTZ。最初 AOT 的 outer FMA 未显式 FTZ，未上 GPU，完整留在 `initial-aot-no-explicit-outer-ftz/`；当前版本已显式使用 `fma.rn.ftz.f32`。CPU 模拟对 outer FMA 的输入和结果都执行带符号 FTZ。原 UMMA K128 内部具体 FP32 求和树仍未知，不把自己选择的 shuffle 树称为原树。

## CPU 数值与 ownership

`cpu-result.json` 保留 checkpoint matrix 的 key/shape/dtype/bytes/SHA、oracle SHA、选择的 route/token/column、每 K128 partial bits、最终 FP32/BF16 bits。CUDA_VISIBLE_DEVICES 为空，CUDA 未初始化。

- L0/L19/L39 各选择同一专家的两条真实 route，每条 64 个 FC2 列，共 **384 个 BF16 输出全部等于原生 oracle**。L0 expert205，L19 expert1，L39 expert0；未按输出值选择或重排。
- 另有 **72 个 FC1 raw FP32** 诊断，无原生前量化 raw oracle，不声明 FC1 exact。
- 同一输入、尺度、outer FMA 和 BF16 规则下，serial128 与本 8-lane 树有 47 个 K128 partial bits、23 个最终 raw FP32 差异；四个串行 K32 再二叉相加有 35/15 个差异。三种树的所选 FC2 BF16 都相等。这个对照只证明 CPU 树之间存在求和顺序差异；不能以此保证全域 exact，也不能把将来的 native 差异自动归因于求和树。
- **147,523 项 CPU ownership 断言**覆盖 rows0/1/2、两个固定 K/output shape、K128 每元素唯一覆盖、4-byte 对齐、输出恰写一次及所有 15 native fixtures 的原 expanded route 分组。两两分组每个原 top8 route 恰出现一次，未减少专家。
- 每 subwarp/chunk 的权重请求是连续 32 B；每 warp 为四个不同输出列的四个 sector。两 route 共享权重加载，但更多同专家 route 仍生成多个任务。真实 L0/L19/L39 分别 319/301/297 个 pair-or-single 任务，原 native token8 tiles 是 208/168/146。请求的逻辑权重字节约 1.003/0.947/0.934 GB/层；缓存可能复用重复任务，因此这些不是 DRAM 字节预测。

## 实际 AOT 和指令预算

CUDA 13.4.46，`-O3 --fmad=false --ftz=false -arch=sm_103a`，两个 kernel 都是 **REG32、0 shared/local/stack、0 spill**。这是实际 CUBIN resource，不从源 register hints 推断驻留。无资源配置或 tile sweep。

两个函数都保留 rows2 稳态 K128 循环 PC 0x1280..0x1bf0，共 152 warp instructions，其中仅值解码和未 scaled 乘加就有：

| 每 warp/K128，两 route × 四输出 | 条数 |
|---|---:|
| E4M3 pair → FP16 pair `F2FP.F16.E4M3.UNPACK_B` | 24 |
| FP16 → FP32 `HADD2.F32` | 48 |
| 未 scaled `FFMA` | 32 |
| 上三项之和 | **104** |
| 另有 shuffle / FP32 add / scale FMUL / outer FFMA.FTZ | 6 / 6 / 2 / 2 |

每次上述 warp 工作为 1024 MAC。四十层、B64、top8 的 FC1+FC2 工作量为 **64,424,509,440 MAC**。假设所有 route 都能完美配对、148 SM、每 SM 4 warp instruction issue/cycle、频率取已有 NCU device record 的 2.032 GHz，则仅 104 条解码+FMA 就对应 **6,543,114,240 warp instructions / 5.4393 ms 的乐观 issue 预算**；完整 152 条稳态循环模型为 7.9497 ms。只有 FP32 FMA 的理想峰值模型为 1.6736 ms，漏掉解码因而不能用于支持本方案。

5.4393 ms 是指定编译映射和频率/issue 模型下的乐观预算，**不是实测延迟，也不是所有 GPU 实现的物理下界**。它未计内存等待、归约、metadata、FC1 FP8 epilogue、activation、finalizer、启动和尾部。单 route 循环为 64 条核心指令/512 MAC，单位 MAC 更贵；因此真实单 route 尾部不会改善这个理想配对模型。原生约 4 ms GEMM 是历史 profile 的比较预算，不是最新服务的实测分项。

架构依据：[CUDA Programming Guide, compute capability 10.x](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capability-10-x) 描述每 SM 128 FP32 cores 与四个 warp schedulers、每 scheduler 每周期一条 ready-warp instruction。实际 GPU 时钟变化、pipeline throughput 和 cache 状态未由本 CPU 测试测量。

## 与已失败路线去重

生产 `kernels.py:291–355` 的 small_batch_moe 已将输入量化为 FP8；不能说它从原 BF16 x 直接计算。它按 route 重复权重、把 scale 乘在各项并用全 K tl.sum，FC1 gate/up 经过 BF16，FC2 route 合并边界也不同。本原型尝试 grouped 原 FP8 与每 K128 原尺度累加，未重复它的数学合同。旧 `native-moe-column-tiles-summary.json` 的 36 配置属于原 Fish tensorcore 路径，保留 Fish 数值但相对现用 FlashInfer 约 3.76–4.70% RRMS，三个最佳点仍慢于现用路径；也不是本固定 SIMT 候选的正确性或性能证据。

没有继续实现完整 FC1/native activation/finalizer，因为实际编译预算已经不支持当前目标。没有 GPU 计时、服务/质量验收、性能晋级或阈值改变。独立复核在同级 `native-fp8-simt-independent-review.md/json`；Ruff 执行发现 C401/FURB167 两个非语义风格项，已执行的 CPU driver 保持原 hash。
