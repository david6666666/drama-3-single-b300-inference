# B64 参考吞吐与真实服务差距

目标仍是各并发吞吐至少达到冻结 vLLM 基线的 1.5×，质量界限与原生 top-8
计算不变。C64 目标为 **6765.407 frames/s**。当前完整异步 codec 可行性组为
**4360.917 frames/s**，还需 **55.14%** 提升；下面的账本用于选择优化变量，
不作为停止优化或声称目标不可达的依据。

`64 / 0.008275 = 7734.139 frames/s` 是已测隔离 B64 decode 的满批参考速率。
原始精确时长为 8.275183868 ms，对应 7733.967 frames/s。它既不是硬件理论
上限，也不是服务保底速度：它只代表一个既定状态、上下文和无同时 codec
争用的实测 decode；新的算子仍可能改变这一参考。6765.407 相当于参考速率的
87.47%，当前服务为 56.39%。不能把“已经测到纯 decode 7734”改写为“服务至少
应有 7734”，也不能把服务差距全部称为可删除的开销。

## 最新完整请求账本

| 项目 | 未开启诊断的 codec-private GPU0 | 新 1088 请求异步时间线 GPU0 |
|---|---:|---:|
| 请求成功数 | 1088 / 1088 | 1088 / 1088 |
| 实际 acoustic frames | 102241 | 102285 |
| frames/s | 4360.917 | 4286.213（仅诊断） |
| 完整 makespan | 23.44484 s | 23.86372 s |
| 最后 dispatch 到结束 | 1.39869 s | 1.89020 s |
| 最后 dispatch 后交付的帧 | 4132 | 4421 |
| 达到 500 帧上限的输出 | 3 | 7 |
| 持续接纳时客户端平均 in-flight | 63.9819 | 63.9826 |
| 完成到下一次 dispatch 的平均间隔 | 0.390 ms | 0.373 ms |

未诊断组的输出均值 93.972 帧，中位数 88，p95 149.65，最长 500。最后完成的
请求是 441 帧、低 RMS 样本，仍完整计入吞吐与评分。新时间线最后完成的是
500 帧样本。两组帧数只差 44，但尾部差 0.4915 s，不能把两组吞吐差全部叫作
profiler 开销；它们也不是新的受控性能 A/B。

原始 dispatch 是 closed-loop 客户端发起 HTTP 的时刻，completion 结束于收完
响应，之后还有 PCM 统计开销。客户端持续接纳时的缺槽面积折算成满 C64 时间，
未诊断组仅 **6.24 ms / 22.038 s**。因此客户端没有及时发请求不支持作为主要
原因。in-flight 包含 admission、prefill、codec 和传输，仍不能当作 GPU active。
最后 drain 也不能整段当作可回收损失，它仍交付真实输出。

按当前输出量归一，未诊断组的参考时间为 13.21944 s，目标时间为 15.11232 s，
实际为 23.44484 s；要到目标需节省等价的 8.33252 s。允许的约 1.89288 s
“高于参考时间”预算只是参照，不证明当前必需的 prompt 工作能装入其中。

## 新异步时间线的可加总分解

只分析新 1088 请求异步时间线，排除 warmup，保留所有尾部。CUDA event envelope
为 23.83705 s；模型区间 union 22.19147 s，codec union 6.94100 s，两者重叠
6.06080 s。因此不能把 codec 的 6.941 s 全加到模型时长，也不能直接叫作可隐藏
收益。模型独占区间 16.13067 s，codec 独占 0.88020 s，两者之外 0.76538 s；
后者含主机提交作用域及其他未覆盖工作。全部已记录事件之外为 0.40989 s。
这些均为事件跨度，可能含主机提交空隙，不是 SM 利用率。

相对于本次输出的满 B64 参考时间 13.22513 s，10.63859 s 的差距可严格记成：

| 账目 | 时间 |
|---|---:|
| 纯 decode 超过其实际 request-row 参考份额 | 4.61099 s |
| mixed、teacher tower、首帧和旧请求 tower，扣除旧请求 decode-row 参考份额 | 4.30495 s |
| 初始独立 prefill | 0.05130 s |
| codec 独占区间 | 0.88020 s |
| 模型/codec 区间之外 | 0.76538 s |
| benchmark 墙钟与事件 envelope 边界差 | 0.02667 s |
| 首帧/EOS request-row 修正 | −0.00091 s |
| 总和 | **10.63859 s** |

这是无重叠的算术账本，不是七个可直接相加的提速承诺。mixed 项包含必须执行的
prompt 和 teacher tower；decode 项混合了未满批、上下文变化、争用与提交差异。
按原生 EOS/max_frames 逻辑条件推导，本次有 102278 个 decode-backbone request
rows，11336 在 mixed 中，90942 在纯 decode 中。trace 的纯 decode call rows
只记录 90739，差的 203 是单请求多步 decode；因此全程纯 decode 的 call 平均
55.33、条件推导的 step 平均 49.34、事件时间加权平均 55.99 是不同指标。

持续接纳期间纯 decode 平均 active=60.862，最后 drain 为 16.811。普通 decode
按 call-row 计的 bucket 多余行是 5.162%（计入推导的 B1 多步后 5.151%）；mixed
旧请求部分是 11.080%。这些都不是时间损失率，mixed prompt 容量 padding 尚未
记录，不能假定整个 mixed 计算也浪费了相同比例。

## 优先验证的原生优化

1. **mixed 数据搬运与 prompt 算子。** 新服务中 mixed+teacher 为 5.244 s / 199
   轮。另一份真实 D58/P6、D64P1280、span256 隔离 profile 为 28.322 ms；三次
   profiler replay 的 kernel sum 平均 29.501 ms，含重叠，不能与服务跨度相加。
   该输入的专家 GEMM 约 7.7 ms、卷积 1.259 ms、直接 BF16 copy 1.611 ms、GDN
   norm 0.716 ms，支持检查布局复制与无效卷积 tile。这里只是一个 mixed 形状，
   不外推到全语料；新布局/卷积开关需分别、组合验证并做完整服务质量。
2. **codec 对 decode 的争用与批组织。** 新时间线相同 active=64 时，重叠 codec
   的 48 轮平均 10.199 ms，无重叠的 46 轮为 8.616 ms；active=63 为
   10.511 / 8.915 ms，active=62 为 10.525 / 8.960 ms。仅持续接纳、匹配
   active=57–64 的组间差累计约 1.670 s；上下文和缓存仍未控制，不能称为
   已经定位或保证回收的争用成本。下一步改变一个批策略并复测，而非只统计
   overlap 越多越好。
3. **先核实拆分，再尝试最少 launch。** 新时间线 1304 次 codec_submit 对应
   2291 次 codec graph/copy，平均实际 batch=2.949；既可能来自不同 frame
   bucket，也可能来自同 bucket 的 DP 拆分。历史 80 请求动态回放可复算为
   75 个非空 emit →95 个 frame-bucket 组→99 次实际图调用；合法最少为95次，
   所以该历史回放仅4次是可省的 DP 拆分，不能把957次差额都归因于 DP。
4. **再评估不等待的相近 frame bucket 合批。** 同轮已经 ready、不同 slot 的
   6/8 帧组可先做独立 6+8→8 原型；每行 valid length、fresh 和 position
   必须保持，实际输出按原长度切片。不能将同一 slot 的连续块或不同 generation
   合进同一批，不能改变先后依赖或额外等待首块。需新的 PCM/状态证明。
5. **尾部与请求接纳保持为解释指标。** 客户端已经基本满并发；更细 bucket 和
   尾部优化可继续做，但需要以服务证据排优先级，不用全程低平均 active 代替
   持续服务占用，也不删除低 RMS、长输出或尾部样本。

`DRAMA3_CODEC_SPLIT_POLICY=min_launches` 已作为默认关闭的单变量实现：保持相同
ready rows、frame buckets、ramp、dtype 和 FIFO，按现有合法 captured batch
尽量少发图，不跨迭代等待。默认 `dp` 原样保留。CPU 5765 项合法分区、padding、
顺序、覆盖、缓存与默认策略检查通过，不等于 GPU PCM/状态或服务质量通过。
需要同初始模型/codec 状态和请求轨迹的 GPU 动态对照，再做同源全量 C64 A/B
及全部音频评分；记录 TTFA p50/p95 和 RTF，不能只看图调用变少。

下一轮 trace 需要补充：每次 emit 的 frame bucket/ready 数、实际 split 和 padded
batch、完整 startup cost_ms；codec jobs 的 slot/generation/fresh/真实帧数/位置，
提交序号、依赖事件和队列深度；mixed 的 prompt token sum/capacity/span；decode
steps 与 GPU/CPU 时间对齐锚点。当前 CPU 阶段边界只是按客户端 dispatch 分组，
没有这些字段就明确标记缺失。

复算入口：[recompute_service_ledger.py](recompute_service_ledger.py)，结果为
[未诊断服务账本](codec-private-service-ledger.json)与
[新时间线账本](latest-timeline-service-ledger.json)。
[历史 codec 拆分重建](archived-codec-fragmentation.json)由
[独立 CPU 脚本](reconstruct_codec_fragmentation.py)生成；
[假设与所需指标](assumptions-and-required-metrics.json)保留证据边界。


## 新增完整 rich trace：以实际 steps 和 codec 分组复算

`rich-timeline-service-ledger.json` 对应冻结 ab6da53 + compact/layout/guard 的 1088 请求诊断，全部成功，101601 帧、23.23366 秒、4373.01 frames/s。它是带事件记录的单次诊断，不替代正式配对验收。连续接入阶段 21.59533 秒、97261 帧交付、客户端平均在途 63.9822；最后一次派发后的完整排空阶段 1.63075 秒、4340 帧，均保留。

直接记录的 1622 次 decode 事件执行了 1736 步、90373 请求行，加上 mixed 的 11228 老请求行共 101601 个执行行。相比按照输出帧数/EOS 推导的 101598 个有效行，多 3 行来自 multistep 内 EOS 后仍执行的图步骤。因此旧事件缺少 steps 时的推导不能冒充直接观测。连续接入 pure decode 平均活跃 60.8489，执行行 padding 为 5.17%；初始/尾部导致全程步加权活跃降到 52.06，不能把全程均值当成稳态空槽。

197 次 mixed 涵盖 1079 个提示、183228 token，捕获 capacity 共 208384，token padding 为 13.73%。196 次 span256、1 次 span512。主要形状 D64/P1024 为 81 次、均值 24.03ms；D64/P1280 为 54 次、27.26ms。mixed_backbone 含 teacher tower，4.9001 秒；新请求 first-frame 0.2339 秒；老请求 tower 0.2894 秒。这些是事件跨度，不能简单当作独立可省内核时间。

codec 实际账本现在可核实：1289 次有音频的 emit，按 frame bucket 拆成 2122 组，再经 DP 变成 2265 次 codec 调用。同 bucket 固定就绪请求的最少调用也是 2122，只有 143 次（6.31%）来自 DP 额外拆分；833 个额外组来自 frame bucket 多样性。实际行数 6728，有效帧 101601，frame padding 后 105530、再加 batch padding 共 119072；合计 padding 17.20%。减少调用不等同减少时间，更不保证 BF16 字节一致；改变 grouping 必须单独验证数值和 WER/SIM。

完整 event union：model 21.4850 秒，codec 6.8762 秒，重叠 6.0098 秒，codec-only 0.86645 秒，两类范围之外 0.85748 秒。同活跃数 57–64、同连续接入阶段的 codec 重叠关联仍为 1.66982 秒，仍不是因果证明。相对孤立 B64 参考 7734.14 的 10.09697 秒差额闭合为 decode 残差 4.32853、mixed/teacher/first/old-tower 残差 3.97171、初始 prefill 0.04811、codec-only 0.86645、其余间隔 0.85748、wall/envelope 差 0.02469 秒；闭合误差约 4e-15。残差含必要工作，不是可回收预算。

本次固定输出帧数要达到 6765.407 frames/s，需要 15.01772 秒，即还需减少 8.21594 秒。优先做固定输入/状态的 decode 与 codec 并行对照（含 stream priority 和双方完成时间），把关联变成可检验原因；其次验证同轮不同 frame bucket 合批的实际算量与状态/PCM；继续针对真实 mixed 形状优化 MoE、GDN 和 dense。目标保持，不把这份账本当成不可达到的结论。

## 完整 layout 配对：吞吐仍未达标

`layout-control-service-ledger.json` 与 `layout-combined-service-ledger.json` 分别有 101130 / 103260 帧，23.13450 / 23.97664 秒，4371.39 / 4306.69 frames/s。候选输出多 2.11%，全程吞吐低 1.48%，因此该单对没有证明端到端收益。候选最后一次派发早 0.25011 秒，但排空多 1.09157 秒（1.19102→2.28259 秒），不能把全程差全部归因内核变慢，也不能删除尾部挽救结论。候选 1088 质量评分 WER 1.875576%、SIM 63.8682；control 1.599263%、64.35488，满足当前相对界限。单次质量通过不等同完整目标成功。

## Min-launch 首次动态检查：保留失败，继续定位数值原因

`codec-minlaunch/result.json` 保留最初严格门槛：同一个 80 请求、414 条动态 trace，DP/sync 为 99 次 codec 调用，min_launches/sync 为 95 次；模型 codes/roster/完成顺序相同，但 5 个请求 PCM 和 20 个 codec state tensor 存在差异，原始 bit-exact gate **FAIL**，后续 async/service 未执行。差异可来自合法 batch shape 的 BF16 rounding，也可能是状态问题；此时尚不能区分。`numeric-result.json` 是独立的数值诊断，沿用原 DP cost table，保存 PCM、真实槽与 scratch 分离的误差、首个改变组的输入/状态，不覆盖原始失败。即使最终属于同操作舍入差异，也必须按已确认的完整 WER/SIM 界限验收，不能直接晋级。
