# Drama 3 高并发差距复盘与原生计算优化路线

本报告区分已测事实、源码事实和待验证假设。最初结构审计对应独立框架
`e58c8418005af8972d1616d5368009bc07dbe651`，本轮 mixed 服务对照对应
`a25c68da0b501374a0f457f4f912543f9cbd821d`；vLLM-Omni 基线固定为
`d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`。源码链接固定到相应审计版本。
mixed-backbone 已完成分层/状态检查和一组全量 1088 请求服务、音频质量对照，
仍低于 vLLM 基线，状态为 **INCONCLUSIVE**。单次结果不构成重复性能结论。

目标保持不变：B300 单卡，C=1/2/4/8/16/32/64 每档吞吐均至少为新 vLLM
基线的 1.5 倍；全量三次重复的 WER 相对同并发基线增加不超过 1 个百分点、
SIM 下降不超过 2 分。保留原生全部 top-8 专家，不新增有损量化。

## 1. 差距在哪里

最新 GPU6 同源、同步 codec 的单次完整 C64 对照为 3816.47 → 4094.18
frames/s，mixed 候选 WER 1.583%、SIM 64.286，双方各 1088 个输出均已评分。
观察到的吞吐差为 +7.28%，但生成帧数和尾部长度也不同，不能归因成同等计算量
下已证实的提速，更不是三次重复验收。C64 vLLM 基线为 4510.27，目标为
6765.41 frames/s；4094.18 为基线的 90.77%，距目标还需要约 65.24% 的提升。
完整指标及混杂因素见第 5 节；此前 GPU0 的 3596.20 → 3643.99 对照继续保留。

独立框架真实 B64 decode 诊断约 8.3 ms，上游服务中的 24 个纯 B64 图
跨度中位数为 9.458 ms。两者不是同一服务控制条件，不能直接宣称算子领先，
但足以否定“独立框架高并发慢必然因为每一步 decode 更慢”的简单归因。
完整请求还支付 prefill、首帧 tower、音频解码、结果回传及队列衔接的成本。

证据：[当前进展](scale150-progress.md)、
[mixed 完整控制](../evidence/20261016-mixed/results/mixed-control-gpu6/benchmark/summary.json)、
[mixed 完整候选](../evidence/20261016-mixed/results/mixed-candidate-gpu6/benchmark/summary.json)、
[此前 GPU0 同步候选](../evidence/20261015-scale150/results/fused-moe-sync-feasibility-gpu0/)、
[真实 B64 decode](../evidence/20261015-scale150/results/fused-moe-input-real-b64.json)、
[上游服务诊断](../evidence/20261015-scale150/results/vllm-service-profile/)。

## 2. vLLM 高并发为什么更好

### 2.1 把新请求 prefill 和已有请求 decode 放进同一次模型执行

vLLM 的 model-state 显式区分纯 decode 和 mixed，构造 prefill 的已知标签、
decode 的历史帧及采样行；随后一次 backbone 处理所有 token，tower 同样
接收混合 token，并只在需要生成的行采样。这样，大型线性投影和 MoE 能将
prefill 行与 decode 行一起执行。优势不只是少一次 Python 函数调用，而是
减少重复权重读取、共享投影和路由工作，避免每次补充请求时完全停下 decode。

源码：[model_state.py:264](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/model_state.py#L264)、
[混合帧/标签准备:326](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/model_state.py#L326)、
[统一 backbone:425](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/fish_omni_model.py#L425)、
[统一 tower:449](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/fish_omni_model.py#L449)。

重构前独立引擎每轮先在主流完整执行 prefill 和首帧，再执行独立 decode 图。
`finish_prefill()` 的 `.cpu()` 等待首帧，decode 后又 `.cpu()` 取主 token；
同步 codec 之后也 `.cpu()` 取 PCM。因此加速某个 decode 小算子只覆盖整轮
的一部分，不能消除补充请求时的串行成本。本轮原型已共享 backbone，tower
与同步音频边界仍分开，不能把旧路径描述当作候选的完整行为。

源码：[主循环:365](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/engine.py#L365)、
[decode 结果回传:443](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/engine.py#L443)、
[首帧等待:580](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/engine.py#L580)。

### 2.2 按 token 预算组织 admission，而非固定最多八个新请求

实际 vLLM 部署的 stage 0 使用 `max_num_batched_tokens=4096`、
`max_num_seqs=128`、chunked prefill 和 `FULL_AND_PIECEWISE` CUDA Graph。
独立框架默认 prefill 上限为 8，只能先补一小组再开始下一个 decode 周期。
两边都有 min-batch=6、min-running=32、max-wait=100 ms 的 admission 策略，
因此不能把“等待凑批”本身说成 vLLM 独有能力。真正差别是凑批后的执行结构
和 token 预算。

一个上游诊断 trace 的 CPU 标注中，mixed 轮实际包含过：
23 个 prefill / 4095 tokens + 1 个 decode、25 / 4073 + 23、17 / 2933 + 47。
这证明上游能在一次调度中超过八个新请求；该旧单波 trace 不代表 1088 请求
稳态的平均分布。新增 256 请求持续补充诊断进一步区分了接纳与排空，见下节。

源码：[上游 admission:337](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/core/sched/omni_ar_scheduler.py#L337)、
[独立 prefill 容量:233](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/engine.py#L233)。
实际配置：本地
`/home/zjy/code/hsliu/artifacts/drama3-b300-20261014/vllm-8933-codes1088/deploy.yaml`。

### 2.3 GDN 将两种状态更新分开，同时保留上层批处理

vLLM 的混合 GDN 路径让 decode 使用 packed recurrent kernel，prefill 使用
FlashInfer，并直接读写 state pool/output；它没有为了共享投影就把单步 decode
强行改成 chunked prefill。独立框架重构应保留这一边界：共同计算投影，再按
两类 token 调用各自的状态算子，最后共同计算输出投影。

源码：[混合 GDN:144](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/gdn.py#L144)、
[跳过不需要的 FLA 元数据:249](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/model_executor/models/fish_omni/gdn.py#L249)。

### 2.4 音频阶段和结果回传有独立调度

实际基线开启 async chunk、两个阶段的 async scheduling、stateful codec，
并由日志确认 private CUDA MPS 已启动。stage 0 runner 是 v2，stage 1 codec
runner 是 v1。基线 codec 为 FP16，本轮独立框架沿用 BF16；两边均采用
6/8/12/16/24/32 首块递增、64 帧主块和 32 帧左上下文。该精度与执行方式
差异必须记录，不能直接将 codec 时长差归因于调度。上游 generation runner 有异步 D2H stream/event 路径，避免把
所有结果处理都放在模型执行的同步边界。

这些是源码和部署事实；现有 trace 尚不能把吞吐差距按 MPS、异步回传和
codec 各自分摊。更不能把仅有异步开关等同于必然提速。

源码：[异步 D2H:75](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/worker/gpu_generation_model_runner.py#L75)、
[异步输出分支:539](https://github.com/vllm-project/vllm-omni/blob/d6ed4f4844fbc41573ffc03247e5b2ff0a38d322/vllm_omni/worker/gpu_generation_model_runner.py#L539)。

## 3. 哪些解释不成立，哪些数字需要修正

- **不是 prefix cache 命中造成基线更快。** 实际部署两个阶段均
  `enable_prefix_caching=false`；两边使用相同预编码参考音频，不能将参考编码
  或离线预计算收益混入推理吞吐。
- **不能拿其他模型配置解释 Drama 3。** `fish_qwen3_omni_high_concurrency_single_gpu.yaml`
  对应另一部署，不能替代本次 `deploy.yaml`。
- **平均 decode batch 49.65 不能解释为稳态一直空闲约 23%。** 384 请求时间线
  共 716 个 decode 记录，其中 533 个 active=57–64，平均 active=61.38。
  active≤32 的 155 个记录跨度合计约 0.949 s，而 full-B64 形状合计约 5.389 s。
  有限请求数的尾部对全局平均有明显影响。仍需明确划分启动、持续补充和尾部，
  才能估算 admission 能回收多少空槽。
- **decode bucket padding 不是现阶段最大的空间。** 按同一时间线的现有
  1/2/4/8/12/16/24/32/40/48/56/64 buckets 重新计算，执行行数比实际 active
  行数多 4.70%；active=57–64 区间为 4.27%。这还不是 GPU 时间损失率，
  细分 bucket 不可能凭此声称几十个百分点的整机收益。
- **piecewise graph 数不等于模型迭代数。** 上游 trace 的某些 mixed 图碎片
  各 54 条，实际对应三个 mixed 轮；同一短 trace 有大量尾部 B1/B2 轮，不能
  作为完整 1088 请求稳态占比。
- **活跃专家比例不是 SM 利用率。** 一次 B64 路由快照平均使用 161.75/256
  专家，按八行 tile 计算的行利用率为 37.4%；这描述 MoE 稀疏布局，不证明
  其余 GPU 时间都可回收，也不授权减少专家。

可复算证据：[非阻塞时间线分析](../evidence/20261015-scale150/results/packed-async-timeline-analysis.json)、
[原始事件](../evidence/20261015-scale150/results/packed-async-timeline.json.gz)、
[路由快照摘要](../evidence/20261015-scale150/results/native-routing-occupancy-summary.json)。
这些均为诊断，不进入性能验收。

### 新增 256 请求持续接纳诊断

新 profile 使用 C64、256 个真实请求，客户端在请求完成后继续补充。
按 CPU `user_annotation` execute-context 作用域计数，共 802 个调度轮：
1 个纯 prefill、37 个 mixed、762 个纯 decode、2 个 empty。PIECEWISE
图碎片不计为独立轮；scheduled prompt 数还可能包含 chunk，不能当作唯一请求数。

| 时间范围 | 调度轮 | 纯 decode 平均 active | mixed 形态 |
|---|---:|---:|---|
| 客户端持续接纳区间（包含初始填充） | 321：1 prefill / 36 mixed / 284 decode | 60.718，范围 52–64 | 平均 7.083 个 prompt 调度条目 / 1179.33 tokens + 54.083 decode |
| 最后排空区间 | 481：478 decode / 1 mixed / 2 empty | 8.793 | 仅剩一个 mixed 轮 |
| 全程 | 802 | 28.146 | 包含初始填充和长尾 |

这修正了单波 trace 无法分析持续补充的局限，也说明不能用全程平均 28.15
推断上游长期低占用。接纳区间仍包含初始大 prefill，不能直接称为严格稳态。
初始三次大 mixed 之后的接纳子集中，33 个 mixed 轮平均为 5.758 prompt /
950.15 tokens + 56.848 decode；这里只描述 refill 结构，不删除任何请求或
用于筛选更有利的服务成绩。因此“最多八个新请求”与上游 token 预算的差别
在初始填充尤其明显；持续补充阶段的优先问题仍是能否共享 backbone/MoE。

接纳区间 mixed GPU annotation 的 union 合计 999.20 ms，纯 decode 合计
3069.41 ms。它们不是 kernel 总时间或 SM 利用率；CPU wall/monotonic
对齐得到的是客户端接纳/排空代理，in-flight 也包括排队和 codec，不能作为
严格 GPU 稳态边界。Profiler 本身影响计时，本诊断不用于验收吞吐。

证据：[持续接纳摘要](../evidence/20261016-mixed/results/vllm-continuous-schedule.json)。
原始 trace SHA-256：
`29e5cafe329cb85eaba69db7536a28a83fb7e3ac29bc154fc8747f317f6a4b3f`；
request metadata SHA-256：
`bd50166976a4753b33c8aeb5a774ebb5711c74ccbba0df1843885f353ecdc996`。
旧 trace 和全部尾部数据仍保留。

### 原生专家 GEMM 的硬件计数器

在真实 B64 输入、一次 decode 位置上采集第 0、19、39 层的 FC1/FC2，共六个
原生 FP8 专家 GEMM。全部保留 top-8、原权重/scale 和 BF16 dense 路径；源版本
`7a8cea1` 与快照模块 hash 一致。三个层的重算活跃专家数为 204、163、116。

| 层 | FC1/FC2 DRAM 读写吞吐（TB/s） | FC1/FC2 DRAM 峰值百分比 | FC1/FC2 tensor pipe 活跃百分比 |
|---:|---:|---:|---:|
| 0 | 5.792 / 5.185 | 75.54 / 67.69 | 3.88 / 3.53 |
| 19 | 5.564 / 4.757 | 72.57 / 62.10 | 4.07 / 3.46 |
| 39 | 5.097 / 4.320 | 66.53 / 56.36 | 4.48 / 3.79 |

实测 DRAM 读取字节数与活跃专家逻辑 FP8 权重字节数之比为 1.00052–1.00138。
这六个样本支持“权重搬运比 tensor 算术更限制这些 GEMM”的解释；低 tensor
活跃率不代表有二十倍可直接利用的算力。SM 有 warp 在途的周期占比为
83.79%–92.25%，也不等于有效算术利用率。活跃专家由相同 BF16 logits 重算
`topk`，未证明 tie 顺序与 FlashInfer 完全相同。

这些是三个层、一个位置的隔离计数器，不能外推成所有层均值或端到端瓶颈占比。
Nsight replay 可串行化 PDL 并改变缓存历史，不能将这里的 kernel 时长与服务
trace 直接比较。证据：[NCU 摘要与限制](../evidence/20261016-mixed/results/ncu/summary.json)。

## 4. 量化空间与收益上限

### 重构前的异步诊断

此前 384 请求诊断窗口为 9.679 s，prefill 事件跨度合计 2.436 s，decode 6.612 s，
codec/copy 2.308 s。codec 与 decode 的重叠已经达到 1.935 s，不能把 codec
的全部 2.308 s 再加为可消除的串行时间。CUDA event 跨度可以包含主机提交
空隙，不能视为 kernel busy time 或 SM 利用率。

同一个 active≥57 子集中，无 codec 重叠的记录平均跨度 9.026 ms，有重叠的
为 10.713 ms；它们的 active、context 和执行时刻不同，这只是资源争用的
候选信号，不能据此断言异步 codec 使 decode 慢了固定百分比。

可以使用两个预算帮助判断方向，但不能当成预测值：

1. 假设把该诊断中全部 prefill 跨度删除且其他工作完全不变，Amdahl 理想因子
   只有约 `1 / (1 - 2.436 / 9.679) = 1.336`。真实重构还要做 prefill，
   因此单靠隐藏原有 prefill 时长不足以解释当时服务所需的 85.66% 提升，
   也不能将这个旧窗口的比例直接套用到新 mixed 候选。
2. 6765.41 frames/s 等价于满 B64 每 9.460 ms 产出一帧/请求。当前 B64
   融合输入纯 decode 诊断为 8.275 ms，理想满批上限约 7734 frames/s；
   达标需要把相当于纯 decode 预算的 87.48% 用在有效生成上，剩余约
   1.185 ms/等价满批周期还要容纳其他成本。这不是实际平均迭代时长，
   也没有计入 tail、prefill、首帧和音频，只说明目标很紧。

所以接下来应同时减少分离阶段的重复工作，并提高混合批的矩阵/MoE 效率。
把已有单算子再加速一两个百分点，无法承担当前缺口。反过来，原生计算约束
下仍不能预先承诺 mixed batching 一定能实现 1.5×，必须用端到端实验确认。

### mixed 候选的连续服务时间线

源版本 `a25c68d`、GPU2 的 384 请求诊断按原始请求时间排除 warmup，保留全部
384 个样本。请求窗口 9.998 s，其中持续接纳区间（含初始填充）8.027 s，最后
排空 1.972 s。所有选中事件都在主流 0，事件跨度合计 9.717 s，其外墙钟时间
约 0.281 s。这些跨度包含可能的主机提交空隙，不是 kernel busy time。

| 阶段 | 全程调用数 | 全程事件跨度（ms） | 持续接纳区间的形态 |
|---|---:|---:|---|
| 独立 prefill | 2 | 51.19 | 初始批 1、8，共 9 个请求 |
| 纯 decode | 716 | 6059.82 | 477 轮，平均 active 61.071，平均跨度 9.049 ms |
| mixed backbone + prefill teacher tower | 69 | 1968.71 | 68 轮，平均旧请求 55.5，平均跨度 28.564 ms |
| 新请求首帧 tower | 69 | 82.23 | 平均新请求 5.456；全程混合接纳共 375 个请求 |
| 旧请求 decode tower | 69 | 100.93 | 对应同批旧请求 |
| 同步 codec | 484 | 1454.24 | 411 批，平均 5.319 条请求流 |

纯 decode 在排空期平均 active 仅 14.025，全程平均 45.367；这不能解读为
持续服务只使用约七成槽位。上游持续接纳诊断的纯 decode 平均 active 为
60.718，与这里 61.071 接近；两次诊断的负载规模和 profiler 不同，不能据此
做受控速度比较，但没有证据支持“主要差距来自持续补充时大量空槽”。

`mixed_backbone` 标注包含 packed prefill 的 teacher-forced tower；不能把
1968.71 ms 全称为 backbone 成本。`first_frame` 的 CPU 记录约 2030.86 ms，
主要等待同流中先提交的 mixed 工作，并非可额外删除的独立 CPU 开销。通用
时间线分析器只识别异步 `codec_gpu_and_copy`，其 codec union=0 不适用于这次
同步 `codec`，阶段摘要明确保留了后者。

同步 codec 占请求窗口约 14.54%；假设全部无代价隐藏，理想上限约 1.170×，
仍不是预测收益，也不足以单独补齐 65.24% 的目标缺口。旧请求 tower 为
100.93 ms，占窗口约 1.01%；仅合并这部分 tower 的收益预算有限，暂不优先
引入新的混合标签与采样次序复杂度。解除首帧 CPU 等待最多回收部分提交空隙，
不能把其 CPU 等待时长再加为收益。下一步先收紧 mixed prompt span，再以动态
轨迹验证 codec 调度。证据：[分阶段摘要](../evidence/20261016-mixed/results/mixed-timeline-phase-analysis.json)。

## 5. 重新优化的优先级

| 优先级 | 路线 | 可验证收益来源 | 先做的最小实验 |
|---|---|---|---|
| 已完成可行性 | 混合 backbone，共享投影/归一化/MoE，attention/GDN 分段 | 同源单组完整质量通过，吞吐仍低于基线；不能宣称正式 +7.28% | 保持 opt-in，保留完整输出与数值/状态检查 |
| P0 | 收紧 mixed packed prefill 的有效 span/chunk 描述符 | 少启动不含有效 token 的卷积/GDN 工作；三形状诊断已精确通过 | 已集成 span 256、长 prompt 回退 512；完整服务及质量均完成，仅单次小幅差值 |
| P0 | 基于请求生命周期的 codec 批处理与异步回传 | 当前同步 codec 约 1.454 s；先定位已有动态质量失败 | 同一动态轨迹逐步核对 codes、状态与 PCM，保持首块 6 帧和输出顺序 |
| P1 | 按 token 预算补充请求 | 主要改变初始填充和大 prompt 调度；持续 refill 平均规模已接近上游 | 固定到达序列和 token 预算，与 mixed/span/codec 分开改变 |
| P1 | 针对 mixed 的算子布局、融合及更少状态拷贝 | 权重流量和无效 prompt chunk 有实测依据，仍需完整 mixed profile | 使用真实输入；不重复已失败的宽列 tile、BF16 投影 tile、关闭 PDL |
| P2 | 统一 tower 或延后首帧同步 | 旧 decode tower 仅约 1% 窗口；CPU 等待主要依赖前序 GPU 工作 | 只有收益预算充分时扩大混合标签、采样和生命周期验证面 |
| P3 | 更细 decode buckets 或 Rust engine | 现证据只支持有限 padding/主机开销空间 | 新 profile 证明收益后再做；不以语言重写替代调度分析 |

其中 packed 图默认 `max_length=512`，GDN 对每个序列保留八个 64-token chunk
描述符，batch=8 加尾 padding 共 72 个；大量真实 prompt 不需要这么多。
intra kernel 对空 chunk 有 early return，其他路径仍有 masked work，因此
不能把描述符比例直接折算为时间收益。混合执行完成初次服务验证后，该方向
已有独立诊断：`d6928fc` 的 D58/P6、D40/P3、D12/P1 共 339 项输出与状态
检查精确通过，512 → 256/192 span 的中位数分别为 30.934 → 28.359、
22.449 → 19.827、19.334 → 16.806 ms。每次 refill 约 2.53–2.62 ms 的差值
仅是诊断；尚无端到端收益结论。证据：[span 检查](../evidence/20261016-mixed/results/mixed-span.json)。

源码：[packed 图默认跨度:101](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/packed_prefill.py#L101)、
[GDN 静态描述符:54](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/gdn.py#L54)。

### mixed 原型的约束

第一版先保留分开的 tower，缩小数值和采样验证面。共享输入/输出投影；
prefill 的 GDN/conv 只重置新请求，decode 的 recurrent state 保持原生命周期；
attention 分别使用 packed causal prefill 和已有 KV-cache decode。混合 backbone
输出后，再处理 prefill teacher forcing、首帧和 decode tower。

代码默认 prefill 为 native MoE、decode 为 FlashInfer；本轮同源服务两臂都显式
使用 `DRAMA3_PREFILL_MOE=flashinfer`，避免把更换 prefill MoE 后端混入比较。
全部 top-8 不变并不等于输出比特不变。必须先做同输入算子检查，并将实际
后端明确记录，不能把整个网络漂移当成
“GPU 精度正常”跳过。共享 BF16 GEMM 的 M 变化也需要同样核查。

新请求首帧生成后再进入下一轮 decode，可以避免同一轮内隐含的额外状态更新。
但这会改变批次和随机数消费次序，质量仍须全量评分。固定回放测试应控制
`sampling_step`、batch row 顺序和执行顺序；不能依靠不同随机样本证明正确性。

所有真请求 slot、decode padding slot、packed padding slot 必须无交集；
重放后比较未参与 slot 的状态，覆盖 slot 重用、部分批、空序列、EOS、
max-frames、context 边界。多种 graph 的临时池不能在有重叠生命周期时复用。

### mixed 原型目前验证到哪一步

初始 D58/P6 通过 730 项分层、graph 和状态归属检查；扩展 D58/P6、D40/P3、
D12/P1 通过 2190 项检查，后两个形状包含 slot 重用，未参与 slot 要求精确相等。
两次检查的覆盖有重叠，不能相加成 2920 个不同案例。它们只验证当前算子/状态
边界，不能替代全部生成音频的 WER/SIM。

扩展诊断期间物理 GPU0 出现外部进程，因此该次计时不具备性能比较资格。
数值结果与原始计时均保留，但不从计时推导提速。证据：
[扩展数值与状态检查](../evidence/20261016-mixed/results/mixed-layers-expanded-v2.json)。

### 同源完整语料服务与质量

GPU6、源版本 `a25c68d`，两臂只切换 `DRAMA3_MIXED_BACKBONE`，均为同步 BF16
codec、FlashInfer prefill MoE、FLA prefill GDN、BF16 recurrent state、原生融合
MoE 输入。两臂各运行一次 C64、1088 请求；使用相同语料与客户端 SHA，保留所有
输出、失败和离群值。这里不是正式物理 GPU0 的三次重复验收。

| 指标 | 分开 backbone 控制 | mixed 候选 |
|---|---:|---:|
| Throughput（frames/s） | 3816.466 | 4094.177 |
| TTFA p50（ms） | 140.674 | 147.565 |
| TTFA p95（ms） | 291.234 | 267.444 |
| RTF mean | 0.346050 | 0.331417 |
| Makespan（s） | 26.9742 | 24.7088 |
| WER（%） | 1.607636 | 1.582517 |
| SIM | 64.48894 | 64.28628 |
| 成功 / WER 已评分 / SIM 已评分 | 1088 / 1088 / 1088 | 1088 / 1088 / 1088 |
| 实际输出 acoustic frames | 102946 | 101162 |

双方单组质量均满足 C64 WER ≤2.950934%、SIM ≥61.881589 的界限。候选观察到
+7.28% frames/s，但输出帧数少 1.733%；达到 500 帧上限的请求由 4 个变为
1 个。相同 sample 中 537 个更短、517 个更长、34 个相同，不能假定计算量
完全相同。最后一次请求发出到全部结束的 drain 从 1.9016 s 变为 1.1362 s，
这 0.7653 s 约占 makespan 差值的 34%。全部尾部和离群样本计入结果。

采样 counter、batch row 与执行顺序会随调度/graph 准备变化，可能改变随机
轨迹；这解释了为什么必须保留输出工作量并完整评分，不是已经定位了音频
差异的根因。完整服务日志没有 admission batch 记录，不能用另一份 384 请求
时间线代替这 1088 请求对照的实际调度分布。

证据：[控制指标](../evidence/20261016-mixed/results/mixed-control-gpu6/benchmark/summary.json)、
[候选指标](../evidence/20261016-mixed/results/mixed-candidate-gpu6/benchmark/summary.json)、
[控制完整质量](../evidence/20261016-mixed/results/mixed-control-quality.json.gz)、
[候选完整质量](../evidence/20261016-mixed/results/mixed-candidate-quality.json.gz)。
当前尚未提名全并发胜出配置。

后续 span 单变量完整服务对照已在物理 GPU0 完成（`9247e4b`），两臂均已开启
mixed，分别使用 span 512 与 span 256/长 prompt 回退 512。控制/候选吞吐为
3997.02 / 4082.76 frames/s，TTFA p50 为 146.182 / 147.932 ms，p95 为
263.725 / 276.424 ms，RTF mean 为 0.329850 / 0.323508。该单次 +2.15% 差值
伴随 p95 退化。双方各 1088 个输出的 WER/SIM 均已评分，控制/候选 WER 为
1.950934% / 1.833710%，SIM 为 63.75201 / 63.62432，单组质量通过，但吞吐
仍低于基线，不能宣称完整验收通过，也不能与 GPU6 的差值相乘。
证据：[span 控制](../evidence/20261016-mixed/results/span-control-gpu0/benchmark/summary.json)、
[span 候选](../evidence/20261016-mixed/results/span-candidate-gpu0/benchmark/summary.json)、
[span 控制质量](../evidence/20261016-mixed/results/span-control-quality.json.gz)、
[span 候选质量](../evidence/20261016-mixed/results/span-candidate-quality.json.gz)。

本轮 mixed prefill GDN 仍为 FLA。历史 FlashInfer GDN 适配器已做边界和独立
FP32 recurrence 检查，但完整服务仅 3465.39 frames/s，因此当时未选用；不能
仅因 vLLM 使用 FlashInfer 就宣称切换后必然更快。任何 mixed 专用重试应单独
冻结后端、状态和质量变量，见[历史结果](full-corpus-results.md)。

## 6. 异步 codec 质量失败：现阶段没有定位根因

已有完整输出的异步同源对照 WER 为 8.047%（控制）和 3.266%（融合），
两者都超过 C64 上限 2.951%。它们不能作为合格性能结果。先前另一次异步
试验 WER 2.462% 通过，也不能覆盖后续失败，更不能挑选较好一次作为赢家。

只读审计确认了若干现有保护：

- codes/rows 在主流 gather 后记录 event，codec 流 wait，并保留 tensor 生命周期。
- codec 提交在一个 FIFO CUDA 流；旧请求最后一块先于新请求同 slot 的 fresh 首块。
- graph 输出通过 `to(float32, copy=True)` 保存，避免下一次 replay 覆盖同一输出。
- codec 与 packed prefill 分别创建 graph pool；现引擎 prefill/decode 本身不并发。

这些保护不能证明整个服务正确，但目前没有证据将失败直接归因为 slot race、
共享 graph pool 或 cuBLAS workspace。后者仍是需隔离验证的假设。

已有精确回放覆盖不足：decode/codec 测试主要固定 B64、24 步、统一换代；
prefill/codec 测试固定六个 prompt、一个 packed capacity，以及 B1/F6、B4/F16、
B8/F64 codec 组合。它们没有复现服务里连续 admission、EOS、不同 bucket、
不同 slot 重用和异步提交交错的完整轨迹。

下一步应保存一段动态调度轨迹：每轮 active slot/generation、admit prompt、
packed capacity、sampling counter、codec rows 与 chunk 长度。分别在同步和异步
执行中重放同一轨迹，依次比较生成 codes、所有模型状态、codec 状态及 PCM。
如果 codes 已不同，先定位 AR/调度；如果 codes 相同而 PCM 不同，才定位 codec。
必要时增加模型与 codec 独立 capture stream 的单变量检查。不得丢弃失败请求。

证据：[精确 decode/codec 回放](../evidence/20261015-scale150/results/decode-codec-overlap.json)、
[精确 prefill/codec 回放](../evidence/20261015-scale150/results/prefill-codec-overlap.json)、
[失败控制质量](../evidence/20261015-scale150/results/fused-moe-input-control-quality.json.gz)、
[失败融合质量](../evidence/20261015-scale150/results/fused-moe-input-feasibility-quality.json.gz)。

## 7. 本轮验收顺序

先冻结当前版本和诊断假设，完成不依赖 GPU 的状态/元数据检查；再做真实模型
单层与完整状态对照。只有通过后，运行一个固定 GPU0 的全量可行性服务试验。
候选有达到全并发目标的实际希望且质量通过，才展开每个并发三次重复和全部
22,848 个输出评分。所有失败、原始时间、命令和 SHA 留存，不调整现有阈值。

最终必须同时报告 frames/s、TTFA p50/p95、mean RTF 与质量；纯图速度、
有效 batch、CPU 等待和 stream overlap 作为解释指标。未通过完整验收前，
状态继续保持 INCONCLUSIVE，OmniRSI README 不增加成功案例。


### 后续混合 MoE 计数器与动态 codec 隔离

真实 D64/P1280、span256 的三个混合层采样中，FlashInfer 实际路由使用
256/238/229 个专家，六个 GEMM 的 DRAM 读取量为唯一活跃专家原生权重的
1.011–1.017 倍。旁路路由记录与原调用输出逐位相同；仍只覆盖三个层和一个
位置。这个结果不支持“大批次重复大量读取专家权重”的假设，因此不重复扫描
先前已尝试的 blocked 布局。证据：[mixed counters](../evidence/20261016-mixed/results/ncu/mixed/summary.json)。

新的固定动态调度覆盖 80 个真实请求、414 条轨迹、slot 复用和 28 种 codec
输入形状。默认 capture 流下，poll 重叠在第 207 条记录首次与串行生成码
分歧，且请求 roster 一致。专用 codec capture 流下，sync/poll/thread 的
生成码、PCM、完成顺序及所有模型/codec 状态逐位相同；两个配置的串行
trace 与 PCM 也一致。只读 ATen 观察器确认：原配置共享一个 32 MiB
cuBLAS workspace，专用流配置的两个地址范围不重叠。

这支持为复现用例隔离 capture workspace，尚不能证明所有历史音频质量失败
均来自此处。验证脚本额外读取完整输出与 metadata，会改变主机提交时间；
完整异步服务质量仍需独立检查。功能默认关闭，使用
`DRAMA3_CODEC_DEDICATED_CAPTURE_STREAM=1` 开启。
[动态对照及限制](../evidence/20261016-mixed/results/codec-dynamic/summary.json)。


GPU0 后续同源异步服务对照已经完成（`7618212`，每臂完整 1088 请求）：

| 捕获工作区 | frames/s | TTFA p50 / p95 ms | RTF mean | WER % | SIM |
|---|---:|---:|---:|---:|---:|
| 共享（控制，质量失败） | 4307.662 | 146.686 / 285.664 | 0.309293 | 6.162606 | 63.124403 |
| 独立（候选） | 4360.917 | 143.351 / 278.698 | 0.307843 | 1.590890 | 64.191350 |

双方均保留全部输出。候选通过单组质量界限，但仍低于 4510.27 基线与
6765.41 目标，缺少所有并发三次重复的验收。共享控制的质量失败意味着不能
把它的吞吐作为可用配置来宣称收益；这次修复的直接证据是固定重放正确性和
完整音频质量恢复。新的引擎入口要求异步 codec 显式启用专用 capture stream，
防止继续使用已复现损坏的配置。历史共享配置仅保留为冻结对照。
[完整汇总](../evidence/20261016-mixed/results/service-feasibility-summary.json)。
