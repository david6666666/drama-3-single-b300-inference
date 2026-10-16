# FC2 原生组件：完整路由扩展验收

结果：**EXPANDED_NATIVE_FC2_EXACT_PASS，306/306 项，15/15 cases**。不包含性能测量，也不代表完整 MoE、驻留 cluster 或服务端优化已经成立。

## 覆盖和不变量

固定 source `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`，checkpoint `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`。使用既有 15 组不可变 native oracle：layers0、19、39，各自 real / left_only / right_only / rows_reversed / worst_global288。

所有 256 专家、原始 top8 和路由权重保留。每 case 的 512 条 expanded routes 使用原始 `expanded_idx_to_permuted_idx`、CTA expert ids、绝对 mn_limit、ncta、padded；CPU 验证 mapping 为单射、expert 一致、每 tile 有效行连续。覆盖 inactive 专家、每专家 1–8 条、跨多个 tile 和最大 64 条，ncta 范围 109–288，缓冲最大 2304 行。

输入是 **原生 activation** 的 E4M3 bytes / FP32 scale，与原始 down_proj 权重。没有使用 FC1 或 activation 候选输出。实际 BF16 输出在 15 cases 的 eager 和 graph 中直接与原生 oracle 相等：每次 512×2048 个值，共 **31,457,280 个 BF16 值比对**。没有校验器重排、容差或筛选。

每层保留同一个 CUDA Graph，依次切换全部五种真实 metadata/activation 分布，包括 padded stride 和 ncta 变化。检查 padding/tail 输出仍为预先写入 sentinel；所有输入/metadata/其他中间缓冲、全部权重和 CUDA RNG 不变。GPU7 已释放。

## 资源和冻结

独立 v2 的唯一改动为 BF16 TMA shared-store 的逻辑行修正，详见 `../expert-fc2-native-v2/epilogue.patch`。MMA64×8×32、K128、原始 FP32 block scaling 和 BF16 边界不变；六类 A/B/scale stage 深度固定4。

CPU shape `[2048,2304,512,256,64]` 生成的 CUBIN 与单专家相同，GPU 三层均匹配 SHA256 `127abe920626d6db37294b3f902e539028de62bb37747a86947622f0a3660358`。共享内存仍为 74,664 字节，小于80KiB；STACK/LOCAL0。CPU preparation 不初始化 CUDA。

启动前 `protocol.json` / `frozen-inputs.json` 冻结脚本、源码、15个oracle、所用checkpoint shards及CUBIN。`process.json`、`gpu-before.txt`、`gpu-after.txt`保留进程和设备审计。原始首轮 FAIL 在 `../expert-fc2-native/` 保留；本次是新协议下修复后的独立结果。

## 与未来驻留实现的接口

1. FC2 logical activation：每8-route tile为 E4M3 `[8,512]`；scale `[4,8]`，第一轴为K128块。全局 padded 模式底层 scale 是一维 buffer，其有效区域按 `[4,padded]` 解释，不能固定2304 stride。
2. activation agent 的四 rank 输出 `[8,128]` + FP32 `[8]` 可在逻辑上合成该输入。该陈述是格式相容性，**不等于当前全局 load/TMA kernel 已可消费 DSMEM**。
3. 每个 FC2 output chunk 保留 K block 顺序0..3和每块4次K32 MMA；原始 BF16 边界、route weighting 与 top8 reduction 后续都必须独立验收。
4. BF16 epilogue 必须把 TMEM 配对值写到逻辑相距8的行后按原生swizzle/TMA发布，不能延续旧相邻vector2 store。当前仅验证 bias=0、非gated FC2 合同。
5. 驻留路径需要明确 FC1 publish→activation publish→跨rank读取→所有reader完成→scratch复用的阶段。现有组件通过没有证明跨CTA内存可见性、barrier phase或slot复用正确，也没有验证完整cluster的实际共享内存/寄存器资源。
6. 当前每CTA **74,664字节** 仅属于独立 FC2 kernel，不可直接与FC1资源相加或当作整体驻留资源预算；必须做buffer生命周期规划和编译后实测资源审计。

入口文件：`probe.py`（明确注明复用 sibling FC1 metadata harness）、`prepare.json`、`result.json`、`protocol.json`。未修改运行时或共享venv，未计时，未宣称吞吐提升。
