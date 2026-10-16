# 原生 BF16 dense GEMM CPU 审计

结论：**有一个尚未执行的明确库算法选择方向，但现有证据不能承诺 ≥0.3ms/step 收益。暂不授权重复 generic backend/tile sweep，不把旧 profile 的25%+6%当当前成本。** 本审计仅 CPU 读取；runtime、算术和公共文档均未改。

## 实际调用与去重

- `src/drama3/kernels.py:274`：B1为专用Triton GEMV；特定B2–32形状为FI `mm_bf16(... backend="cute-dsl")`；B64及mixed大token batch落到 `torch.nn.functional.linear`。因此当前 B64 没有自定义 dense算子选择表。
- `model.py:281` 的 `DRAMA3_AUTOTUNE_CACHE` 只调用 MoE warmup，不能称 dense cuBLASLt autotuning。源码未找到显式 Lt algo ID/tile/split-K/reduction/workspace选择。
- `scripts/benchmark_projections.py:25` 已扫B1/2/4/8/16/32/64和current/torch/cute/tgv。不要重跑这些 generic backend组合。
- `scripts/diagnostic_bf16_projections.py:34` 与 `bf16_projection.py:13` 已在真实decode输入上扫144个原生BF16/FP32acc Triton组合（9形状×BM16/32×BN64/128×split1/2/4/8），无一个满足选中收益。`native-bf16-projections-b64/projections.json` selected=[]；全部实际weight按模型顺序读的序列1.644203→1.648367ms，没有收益。它的`checks=[]`是没有候选被选中，并不表示逐层某个新算子获得验证。
- `fused_projection.py:40`仅B1 sigmoid/SwiGLU GEMV，保留BF16中间舍入；`table_projection_batched.py`是B2–8特定tower表投影。均非尚未被测试的B64通用GEMM+norm融合。packed QKV、packed shared/router已经存在；重复pack也不是新方案。GDN recurrence+norm三版已失败，不能改名再试。

## 能证实的预算

最早 `drama3-b300-20261014/gdn-tuned-profile-b64/table.txt` 的splitKreduce2.440ms/5replays=0.488ms，与后续真实输入的原生路由/算子分布不同。真实输入 `drama3-scale150-20261015/native-bf16-projections-b64/table.txt` 明确三次replay，splitKreduce665.671us/3=**0.221890ms/step**（201calls/3）。即使免费移除全部该归约也达不到0.3ms，且改为non-split会损失并行，不能视为免费预算。

较可信的历史整投影序列为1.644203ms；要净省0.3ms需整体改善**18.246%**，不能靠一个7us热矩阵的漂亮数值。该序列也不是最新pipeline/codec重叠profile：新服务ledger没有kernel级dense归因，mixed-profile是旧source-initial的58:6/1280token场景，不能搬其百分比。现有CPU证据只证明dense仍有毫秒级总工作，并未证明可优化部分超过0.3ms。

## 唯一可考虑的后续：有界、记录明确的原生 cuBLASLt 算法选择

如parent决定保留此路线，最短准备是artifact-only薄C++/Lt binding与真实矩阵序列接口，预计约45–90分钟CPU实现/编译/契约审查（估计，不是已完成）。不先实施runtime。固定实际B64、stride/alignment、BF16 A/B/D、FP32 compute/alpha、beta0和capture前分配workspace；保留native F.linear最佳基线。只比较已记录Lt heuristic给出的**非split-K**与**FP32 workspace reduction split-K**两个算法家族，最多两种代表形状，以实际完整权重序列累计预算决定是否扩展；不是任意算法/tiles的大扫。

安装头文件 `nvidia/cu13/include/cublasLt.h:1909` 明确NONE、INPLACE、COMPUTE_TYPE、OUTPUT_TYPE；`:2536` SPLITK_NUM与`:2542` reduction scheme。BF16 OUTPUT_TYPE/INPLACE会改变中间舍入，不能默认为原FP32归约。要求记录algo ID、tile、split、reduction、workspace、输入/累加数值implementation flags（:980 FP32、:987 BF16）、实际kernel名；优先NONE或COMPUTE_TYPE，禁止TF32/FP8替代、不能把选择日志当执行证明。

第一步是重新用**当前已冻结原生真实输入的投影调用序列**测总预算，并保存所有实际shape/stride/weight顺序；若可测目标集合总成本与合理改进仍不支持0.3ms，就停。若算法输出非bit-exact，保留原exact FAIL，先用独立FP64 dot-product参考定位为同BF16输入/FP32求和树变化，逐层边界检查、unused/state/metadata所有权仍须精确；不能用WER容忍布局/race/错state。之后才允许受控整图，最后同源完整1088 WER/SIM；单个算子数值小误差不等于服务质量验收。

## 当前模型主流与 backend（供后续架构图使用）

`Engine` 管理admission、slot生命周期及native EOS；mixed graph把新prompt与已有decode的backbone dense/MoE合批，causal attention与GDN状态更新分段，teacher tower/首帧和老请求tower分开。可选两帧图仍按native EOS/slot条件控制，没有少算专家。详见 `mixed_model.py:44,113,189`、`model.py:690`。

Backbone是原生BF16 dense/normalization与GDN/KV状态，MoE全部256专家路由选top8；大batch routed路径为FI TRTLLM原生block-FP8 E4M3权重/激活及FP32每128scale，shared expert仍BF16。GDN decode使用SGLang/FLA packed recurrence Triton，prefill默认FLA/compact变体；attention走FI算子，tower输出原semantic与10-codebook帧。codec接收到保持顺序的私有codes，原statefulDAC图，独立capture stream与异步D2H/callback。codec transformer targeted cached linear是独立默认关闭实验，不能图中画成全codec BF16转换。现有runtime主体是Python/Torch/CUDA Graph，不应把允许使用Rust写成已经部署Rust engine。

本报告没有新增性能结论，也未修改正式验收阈值。
