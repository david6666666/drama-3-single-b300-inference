# FC2 v2：只修正原生 BF16 写回布局

状态：**ONE_EXPERT_FC2_EXACT_PASS，40/40**。原始 `../expert-fc2-native/result.json` 的 FAIL 与所有失败数据保持不变。

本次只在进程内替换 FlashInfer `GmemCResource` 的 `_stage_swap_ab_bf16_dsfp8_tma_pair`，修改见 `epilogue.patch`。TMEM 配对的两个 BF16 值分别属于 `logical_row0=(base_row//16)*16+(base_row%16)//2` 和 `logical_row0+8`。两个值分别按原 `_bf16_tma_smem_element_offset` 计算 swizzle 地址后写 SMEM；原有 conversion、scale、K128 累积、TMA commit、barriers 均保持。

没有在校验器重排输出，也没有量化参数变化。原生 layer0/expert205 8 routes 的 16,384 个 BF16 输出直接与 oracle 逐字节相等；eager 与同输入 CUDA Graph 均通过。19 项其他输入、权重、元数据和 CUDA RNG 不变检查在两种调用方式下均通过。

CPU AOT 与 GPU 生成的 CUBIN 一致：`127abe920626d6db37294b3f902e539028de62bb37747a86947622f0a3660358`。动态共享内存 72,616 字节 + 静态 2,048 = **74,664 字节**；STACK/LOCAL 0。CUBIN 基础 REG30 不代表完整 warp-specialized 动态寄存器占用，不据此声明 occupancy。

`protocol.json` 和 `frozen-inputs.json` 在 GPU 启动前生成；checkpoint/oracle、运行时 source95fcb0e、原始 installed FI 与 CUTLASS 环境保持不变。没有计时或服务 benchmark。

后续全部 15 原生 oracle 的完整专家路由扩展结果见 `../expert-fc2-expanded/report.md`。
