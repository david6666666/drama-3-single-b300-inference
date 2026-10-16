# Metadata-ready FC2：条件性运行时接入设计（CPU 审计，未实施）

**当前停止条件：候选不能进入运行时。** 本轮完整调用 gate 的 1516 项精确检查通过，但三个 real fixture 的单次延迟降低分别为 2.7294%、3.1834%、4.1453%，均低于冻结的 5% continuation 门槛。launcher 已停止，12 个压力场景的性能没有运行。下面只是按父任务要求保存的接入设计，不是推广建议，不改变门槛，也不声称服务收益。独立复算见 `metadata-ready-independent-screen-review.json`，16 项检查通过，包括 130 个冻结文件 hash。

审计运行时为 `e347d15f872e685f66b2cecf37318a4565fd8f8f`，分支 `codex-drama3-scale-150`；源码目录 `/home/zjy/code/hsliu/drama-3-single-b300-inference`。未改 production、共享依赖或已执行实验文件，未运行 GPU。

## 现状与最小范围

- `src/drama3/model.py:520–584` 的 `routed_moe` 对适用 batch 先沿原路径生成 E4M3 activation、FP32 block scales 和 FP32 router logits，再调用 `trtllm_fp8_block_scale_moe`。`model.py:543–555` 已有原生输入融合；应保持它和原始 fallback，不能改成归档 expert IDs 或 Python `topk`。
- `model.py:195–205` 已将 checkpoint `[gate|up]` 转为 FlashInfer `[up|gate]`；候选只接受当前已验证 MajorK 权重。不要再次换序、量化、改变 top-8，或把 BlockMajorK 当成同一 ABI。
- `model.py:595` 的 backbone 遍历 40 层。tower 没有此 MoE 候选；first-frame、prefill、mixed 的行数/状态合同不属于当前 B64 specialization。
- `engine.py:60–136` 的 `DecodeGraph` 目前构造普通 `CUDAGraph()`，warmup 后捕获整个 forward。第一版最小接入只允许该类 `size==64, first_frame==False, steps==1`，原生 PDL 开启、MajorK、单主流、完整 256 专家/8 routes/2048 hidden/512 intermediate。所有其他条件走原有实现。`Engine` 在 `engine.py:358–388` 构造多种图，不能只凭 `len(x)==64` 启用。
- `FramePipelineGraph` (`src/drama3/frame_pipeline.py:20–115`) 有两个 backbone、辅助流和 tower overlap，`MixedGraph` (`mixed_graph.py:16–117`) 有共享 prefill/decode 行。这两类最初 fallback；后续需要各自独立的 80 个调用/跨流或 mixed-state 证明，不能继承普通 40 层 graph 的 PASS。

## 从真实 logits 到当前 body 的 installed API

以下均是当前安装的 `flashinfer.fused_moe.core`；它们不是只依赖公开版本号即可假定稳定的跨版本接口。安装路径前缀为 `/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/flashinfer`，源码/hash 应随 adapter 固定。

1. `core.py:4238` `allocate_trtllm_moe_canonical_routing(logits, top_k=8, tile_n=8)`：在非 capture 的准备阶段分配 11 个稳定 Tensor。
2. `core.py:4260` `canonicalize_trtllm_moe_routing_(canonical, logits, None, quantized, ...)`：每次 forward 读取**当次真实 logits**。参数保持 `top_k=8, n_group=None, topk_group=None, local_expert_offset=0, local_num_experts=256, routed_scaling_factor=1.0, routing_method_type=4, use_routing_scales_on_input=False, use_deep_seek_fp8=True, norm_topk_prob=True`，PDL 沿用生产解析后的值。
3. `data/csrc/trtllm_fused_moe_kernel_launcher.cu:6271–6332` 证明该接口调用生产 `Routing::Runner::run`，而不是另写 softmax/topk；产出原生 int16 IDs 和 BF16 route weights，并写出 tile8 的完整 metadata。它仍需对当前实际 logits 逐位验证；源码调用相同不是端到端精度 PASS。
4. 当前安装完整 FP8 public call 在 `core.py:3000–3003` 解析 `enable_pdl=None` 并应用设备支持限制。adapter 不应把用户的 PDL=0 强制变成 1；不满足当前候选 PDL 合同时 fallback。
5. 较保守、与已执行 q/ids harness 最接近的组合：用 `core.py:4138` `trtllm_moe_allocate_routing_metadata_multi_tile(..., tile_ns=[8], routing_input_mode=UnpackedPrecomputed, topk_weights=canonical.expert_weights)` 预分配，再在每次 canonicalize 后调用 `core.py:4204` `populate_trtllm_moe_routing_metadata_`。这样可直接使用目前已证明的 slot contract，但每次新增一次 metadata pass；这个成本必须纳入真实完整调用，不能隐去。
6. 更少 launch 的备选：canonical 路由本身已产出同 tile8 的 metadata，可建**有名且只读的 Python view**，不再 populate。ABI 来自 `core.py:201–227` 和上述 C++ `CanonicalRoutingBuffers:223–279`：

| canonical.tensors() index | 名称/用途 |
|---|---|
| 0 | routing_replay_ids，仅验证真实原生 top-8 逻辑顺序 |
| 1 | expert_weights，BF16；原 finalizer 输入 |
| 3 | num_tokens_per_expert |
| 4 | total_num_padded_tokens |
| 5 | expanded_idx_to_permuted_idx，activation/finalizer 共用 |
| 6 | permuted_idx_to_token_idx（含 guard element），FC1 route_map |
| 7 | expert_count_histogram |
| 8 | cta_idx_xy_to_batch_idx，FC1/FC2 tile_idx |
| 9 | cta_idx_xy_to_mn_limit，FC1/FC2 absolute mn_limit |
| 10 | num_non_exiting_ctas |

第二种省掉重复 metadata 生成，但引入明确的本地 FFI 布局依赖。只有 shape/dtype/容量、token/expert 双射、每 tile prefix、runtime padded stride、同真实 logits 的边界/最终输出验证全部通过，才能采用；不允许只把旧归档 map 填入新接口。两种选择必须单独记录，不能在同一次实验中静默更换。

执行 body 仍是已冻结的 FC1 → 原 native activation → restricted FC2 → 原 native finalize，最终 top8 累加顺序/BF16 输出边界不变。可复用 installed `build_fp8_block_scale_launch_io` (`prims_ts/moe/tensor_adapter.py:1323`) 和 `_launch_arg_tuple` (`prims_ts/batched_gemm/batched_gemm_run.py:3413`)；restricted FC2 必须经过本地 typed validator/compile 入口，不能绕回通用 kernel 或放宽安装包 validator。

## 外层 40 层图：只捕获一次，在退出 capture 后统一补边

现有 `MetadataReadyGraph` 只认可一个闭合 MoE 图。**不能在 `routed_moe` 内调用其 `capture()`**，也不能把 40 层大图直接交给它的单块全图 checker。

最小结构是新增一个由 `DecodeGraph` 持有的 outer capture owner，复用纯 `Node/Edge/Snapshot` 检查逻辑：

1. 在 warmup/capture 前完成全部资源分配、CUTE 编译和 lazy helper 初始化。`tensor_adapter.py:31` 的 expert-scale-ones cache 等不能第一次在正式 capture 中创建。
2. 仅目标 `DecodeGraph` 改用 `torch.cuda.CUDAGraph(keep_graph=True)`，仍然只运行一次外层 `torch.cuda.graph(...)`。Scoped collector 绑定 graph generation、capture stream、调用 ordinal 和 experts 对象；拒绝重入/混用。普通 eager 调用和其他 graph 类型不取得 restricted launch permit。
3. `routed_moe` 在量化/真实 logits 生成后，将实际 metadata callback、FC1、activation、FC2、finalize 分别交 collector。每段以当前 capture 的真实节点前后差集记录；不能由调用者提交一个 metadata producer 子集。metadata callback 包含 canonicalize 与选定的 populate（若采用）。
4. 同一 layer 的五段节点必须闭合、互不重复，三个 body 相邻 PROGRAMMATIC 边必须仍是 `from_port=1,to_port=0`。所有该次 metadata 节点原先均须先于 FC1。验证可用该段 induced subgraph，但**原始外层图不删除任何节点或边**。所有外层 restricted FC2 节点必须恰好属于一个已登记调用，不能有未登记实例。
5. capture 结束后、instantiate 前，由同一个 owner 给每次调用的**全部 metadata producer**增加 `DEFAULT` full-memory edge 到其自己的 FC2。不能给 activation→FC2 增加 DEFAULT；这会把待测试的 overlap 串行化。保持所有 attention/GDN/quantization/shared-expert/tower/采样原始边不动。
6. 对整个 graph 证明只增加预定的 metadata→FC2 边，节点、kernel launch 属性和原始边完整保留；再检查 40 个调用各自的 validator/完整节点归属，之后才 instantiate。不得重放未封存图。每次 hot replay 只检查 owner/state/generation，不再查询整图或做 CPU 同步。

这里的 capture 快照同现有 guard 一样只证明拓扑和列出的 launch 属性，不 fingerprint 全部 kernelParams/memcpy payload；依靠私有 graph handle、可信 callbacks、禁止外部改参数以及绑定/生命周期检查。不能把这个工程保护层宣传成安全沙箱。

## 缓冲与生命周期：不要把 warmup 的指针捕获进去

- 每个 graph invocation（第一版 40 个）保留 canonical/metadata、FC1 bytes+scale、activation bytes+scale、FC2 BF16、routed BF16、所有 CUDA/TMA 描述符、compiled callable 和 IO `_keepalive`。expert 权重/scale 借用原模型只读存储；不克隆或改 dtype。
- `tensor_adapter.py:1542–1609` 直接把 tensor `data_ptr()` 写入 IO；`1680` 附近的 `_keepalive` 负责存储保活。因此 warmup 的临时 q/scales/logits 指针不能直接拿来正式 capture。函数可以按 config 缓存，但正式 capture 的 IO 必须从那次实际 q/scales/logits 构造，并由 graph owner 持有它们和 `_keepalive`，禁止随后覆盖 IO 字典的指针。
- 准备阶段可预分配恒定 workspace；正式 capture 只做 Python descriptor 绑定和已 warmup 的 launch，不能编译、autotune、调用 `.cpu()`、event synchronize 或分配新持久 GPU workspace。若不能证明绑定路径无 lazy GPU allocation，先停止实现，不以 silent copy/rebind 掩盖。
- 初版每次调用独占 mutable body workspace；不按 layer id 跨 graph/跨帧复用。即使层串行，也不能在没有 reader completion 证明时用共享缓存覆盖前一层的路由/activation。资源节省属于后续独立优化。
- 按 CAP2304 固定容量，五个中间 tensor 加 routed output 约 13,348,864 bytes/调用，40 次约 509.22 MiB，另有 canonical/input/compiler/graph 内存。此数是静态容量预算，不是实测增量；需要对比原 allocator 峰值。两帧图会有 80 次调用和跨流生命周期，不能继承 40 次预算。
- native codec 的独立 capture stream/线程保留。此设计不触碰音频队列、slot 生命周期、EOS、采样 RNG 或 admission。

## 继续前必须验证的内容

当前局部门禁失败，以下均为未执行的未来条件：

1. 相同当前模型产生的所有 40 层真实 q/scales/logits；对照完整 public native FromLogits call。验证真实 router IDs 的原 top8 顺序、BF16 weights、各臂自身 metadata 的 bounds/bijection/expert/token/prefix，以及 FC1/activation bytes+FP32 scale、FC2 BF16、最终 routed 输出。不得按输出值重排。
2. 全部 40 个 outer graph scopes 和补边证据；没有 nested capture、未登记 FC2、失效 descriptor 或 cold allocation；检查所有 owner 缓冲相互地址/范围。pre/post 参数 bytes 和 readonly buffers 不变。
3. 同初始状态、同输入 schedule 的完整 decode graph：B64 的实际 64/63/58 等 live rows、padding slots、不同上下文长度、槽复用、EOS 和跨多次 replay。检查所有活跃及无关 KV/GDN/conv、frame codes、请求 metadata、sampling_step/RNG；不能只检查 MoE 输出。
4. 原生 full graph 的同源配对计时，包含真实 router、metadata 与全部 body/最终累加。局部 q+ids screen 排除了 router 和输入量化，新增 canonical/populate 的代价可能吃掉局部收益。不得将每层约 3µs 的这一轮单次差额直接乘 40 当服务提升。
5. 只有上述通过，才考虑完整 1088 请求性能与全部 WER/SIM；用户确认的 WER+1百分点/SIM-2 分是最终质量界限，不能替代 metadata/state/capture bug 检查。frame pipeline/mixed/其他 batch 需独立支持和证据，不能提前推广。
