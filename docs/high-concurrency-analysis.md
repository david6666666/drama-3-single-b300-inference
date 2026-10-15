# Drama 3 高并发差距复盘与原生计算优化路线

本报告区分已测事实、源码事实和待验证假设。审计对象为独立框架
`e58c8418005af8972d1616d5368009bc07dbe651` 与实际基线使用的 vLLM-Omni
`d6ed4f4844fbc41573ffc03247e5b2ff0a38d322`；下文源码链接固定到这些版本。
本轮正在实现的 mixed-backbone 原型尚未完成验证，不能据此宣称性能提升。

目标保持不变：B300 单卡，C=1/2/4/8/16/32/64 每档吞吐均至少为新 vLLM
基线的 1.5 倍；全量三次重复的 WER 相对同并发基线增加不超过 1 个百分点、
SIM 下降不超过 2 分。保留原生全部 top-8 专家，不新增有损量化。

## 1. 差距在哪里

当前单次 GPU0 同源同步-codec 对照为 3596.20 → 3643.99 frames/s，
后者 WER 1.742%、SIM 64.115，1088 个输出均已评分。这是小幅可行性改善，
不是完整验收。C64 vLLM 基线为 4510.27，目标为 6765.41 frames/s。
3643.99 仅为基线的 80.79%，距目标还需要约 85.66% 的提升。

独立框架真实 B64 decode 诊断约 8.3 ms，上游服务中的 24 个纯 B64 图
跨度中位数为 9.458 ms。两者不是同一服务控制条件，不能直接宣称算子领先，
但足以否定“独立框架高并发慢必然因为每一步 decode 更慢”的简单归因。
完整请求还支付 prefill、首帧 tower、音频解码、结果回传及队列衔接的成本。

证据：[当前进展](scale150-progress.md)、
[GPU0 同步候选](../evidence/20261015-scale150/results/fused-moe-sync-feasibility-gpu0/)、
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

当前独立引擎每轮先在主流完整执行 prefill 和首帧，再执行独立 decode 图。
`finish_prefill()` 的 `.cpu()` 等待首帧，decode 后又 `.cpu()` 取主 token；
同步 codec 之后也 `.cpu()` 取 PCM。因此加速某个 decode 小算子只覆盖整轮
的一部分，不能消除补充请求时的串行成本。

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
这证明上游能在一次调度中超过八个新请求；不代表 1088 请求稳态的平均分布。

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
runner 是 v1。上游 generation runner 有异步 D2H stream/event 路径，避免把
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

## 4. 量化空间与收益上限

384 请求诊断窗口为 9.679 s，prefill 事件跨度合计 2.436 s，decode 6.612 s，
codec/copy 2.308 s。codec 与 decode 的重叠已经达到 1.935 s，不能把 codec
的全部 2.308 s 再加为可消除的串行时间。CUDA event 跨度可以包含主机提交
空隙，不能视为 kernel busy time 或 SM 利用率。

同一个 active≥57 子集中，无 codec 重叠的记录平均跨度 9.026 ms，有重叠的
为 10.713 ms；它们的 active、context 和执行时刻不同，这只是资源争用的
候选信号，不能据此断言异步 codec 使 decode 慢了固定百分比。

可以使用两个预算帮助判断方向，但不能当成预测值：

1. 假设把该诊断中全部 prefill 跨度删除且其他工作完全不变，Amdahl 理想因子
   只有约 `1 / (1 - 2.436 / 9.679) = 1.336`。真实重构还要做 prefill，
   因此单靠隐藏原有 prefill 时长不足以解释从当前服务提升 85.66% 的完整目标。
2. 6765.41 frames/s 等价于满 B64 每 9.460 ms 产出一帧/请求。当前 B64
   融合输入纯 decode 诊断为 8.275 ms，理想满批上限约 7734 frames/s；
   达标需要把相当于纯 decode 预算的 87.48% 用在有效生成上，剩余约
   1.185 ms/等价满批周期还要容纳其他成本。这不是实际平均迭代时长，
   也没有计入 tail、prefill、首帧和音频，只说明目标很紧。

所以接下来应同时减少分离阶段的重复工作，并提高混合批的矩阵/MoE 效率。
把已有单算子再加速一两个百分点，无法承担当前缺口。反过来，原生计算约束
下仍不能预先承诺 mixed batching 一定能实现 1.5×，必须用端到端实验确认。

## 5. 重新优化的优先级

| 优先级 | 路线 | 可验证收益来源 | 先做的最小实验 |
|---|---|---|---|
| P0 | 混合 backbone：prefill/decode 共享线性层、归一化和 MoE，attention/GDN 分段 | 减少分离批的重复权重读取，并让 decode 行搭上 prefill 的较大矩阵 | 从同一真实状态，对比“分开执行一次 prefill+decode”和“混合执行”；逐层输出、所有可变状态及未参与 slot 检查后再测时间 |
| P0 | 按 token 预算补充请求，并允许多个新请求共用一次混合执行 | 减少固定八请求上限和多次首帧/阶段切换；改善补充期调度 | 首先固定完全相同的到达序列，记录 token 数、active/pending/waiting、GPU 阶段时间；与混合算子分开改变参数 |
| P1 | 基于请求生命周期的 codec 批处理与异步回传 | 减少很多小 codec batch、同步 PCM 回传及不必要唤醒 | 先解决下面的动态质量诊断；保持首块 6 帧和输出次序，不能用推迟首音频换吞吐 |
| P1 | 收紧 packed prefill 的有效 span/chunk 描述符 | 少启动不含有效 token 的卷积/GDN工作 | 同一 packed 输入比较静态 max-span bucket，验证 chunk 边界和空序列，再测真实 prefill |
| P2 | 高并发算子布局、融合及更少状态拷贝 | mixed 后重新 profile，针对剩余主耗时优化 | 使用 mixed 的真实输入；不重复已失败的宽列 tile、BF16 投影 tile、关闭 PDL 等实验 |
| P3 | 更细 decode buckets 或 Rust engine | 现证据只支持有限 padding/主机开销空间 | 只有新 profile 证明收益后再做；不以重写语言替代调度分析 |

其中 packed 图默认 `max_length=512`，GDN 对每个序列保留八个 64-token chunk
描述符，batch=8 加尾 padding 共 72 个；大量真实 prompt 不需要这么多。
intra kernel 对空 chunk 有 early return，其他路径仍有 masked work，因此
不能把描述符比例直接折算为时间收益。该方向次于混合执行。

源码：[packed 图默认跨度:101](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/packed_prefill.py#L101)、
[GDN 静态描述符:54](https://github.com/david6666666/drama-3-single-b300-inference/blob/e58c8418005af8972d1616d5368009bc07dbe651/src/drama3/gdn.py#L54)。

### mixed 原型的约束

第一版先保留分开的 tower，缩小数值和采样验证面。共享输入/输出投影；
prefill 的 GDN/conv 只重置新请求，decode 的 recurrent state 保持原生命周期；
attention 分别使用 packed causal prefill 和已有 KV-cache decode。混合 backbone
输出后，再处理 prefill teacher forcing、首帧和 decode tower。

当前 prefill 默认 native MoE，decode 默认 FlashInfer；直接把它们合并为一个
MoE 调用会统一后端并改变原有舍入路径。全部 top-8 不变并不等于输出比特不变。
必须先做同输入算子检查，并将这一变量明确记录，不能把整个网络漂移当成
“GPU 精度正常”跳过。共享 BF16 GEMM 的 M 变化也需要同样核查。

新请求首帧生成后再进入下一轮 decode，可以避免同一轮内隐含的额外状态更新。
但这会改变批次和随机数消费次序，质量仍须全量评分。固定回放测试应控制
`sampling_step`、batch row 顺序和执行顺序；不能依靠不同随机样本证明正确性。

所有真请求 slot、decode padding slot、packed padding slot 必须无交集；
重放后比较未参与 slot 的状态，覆盖 slot 重用、部分批、空序列、EOS、
max-frames、context 边界。多种 graph 的临时池不能在有重叠生命周期时复用。

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
