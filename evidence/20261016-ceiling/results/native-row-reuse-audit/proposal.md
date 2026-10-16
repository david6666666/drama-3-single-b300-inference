# 当前原生空间与逐行共同计算复用提案

本轮建议只做一个新的、小范围可证伪实验：**在 mixed backbone 的 MoE 输入处合并同批、因果前缀相同且实际 hidden 字节相同的行，执行一次原生 `moe_parts`，再展开三个结果到原行序。** GDN、attention、全部 slots、teacher tower 和采样流程保持原计算。先证明实际 hidden 相等，再决定是否值得进入 compact/expand 数值门。没有 GPU 运行，也没有运行时修改。

它与此前 reference-prefix 的 FP32 状态快照/续算方案不同：此处不导出或恢复 recurrent state，不分叉 GDN，不拼接 prefix KV，也不依赖跨批 cache 已就绪。它复用已经证明相同的逐行函数值。旧方案仍可作为更大的后续能力，但其复杂工程和约0.243秒自然就绪命中预算不应再次直接投入。

## 当前服务成本与目标差距

采用 `current-pipeline-service-ledger.json` 的 e7cda1e、1088请求、全尾部诊断，不把更早同步账本当当前下界。该次输出105160帧，22.897230秒、4592.695 fps；固定这些输出达到6765.407需要15.543780秒，即减少7.353450秒。最佳未profile单次4726.77仍需吞吐增加43.13%，等价于固定工作时间下降30.13%。两次运行不能合成为受控性能结果。

| 当前诊断量 | 时间/规模 | 含义 |
|---|---:|---|
| pure decode | 15.803844秒、1035次图调用 | 包含单/双步；调用数不是帧步数 |
| mixed backbone+teacher | 4.946353秒、196次 | 主战场之一，但内含必须执行的旧decode与teacher |
| mixed新首帧/旧tower | 0.230164/0.261235秒 | 原joint-tower方案即使全删也预算有限 |
| model/codec union及交集 | 21.289383 / 6.679043 / 5.709033秒 | codec绝大部分已重叠，不能再加作串行收益 |
| codec only / neither | 0.970010 / 0.613251秒 | 零成本删除也是条件预算，不是可实现方案 |
| decode CPU提交 | 0.072875秒总和 | 不支持语言重写可省数毫秒/step的解释 |
| mixed行数 | 183228真实prompt，209664 prompt容量 | 原padding比真实prompt多14.43% |
| 持续接纳的active | 时间加权约60.252/64 | 不能用全程含排空的平均值宣称大量稳态空槽 |

如果在这个固定账本中把**全部**mixed事件免费删除、其他事件不变，仍仅5858.21 fps。这只是固定轨迹的极端删除预算，既不保证重调度后相同，也不证明硬件不可能达到目标。它说明单独一条只影响mixed的改动不应被宣称能够填补此次7.35秒缺口。

## 已有经验怎样限制下一步

读取了 OmniRSI 的 Strata architecture/expert-residency 与 controlled-tuning记录、vLLM-Omni concurrency/state/overlap记录，以及SGLang-Omni pipeline-state/transport记录。这些是固定来源的指导，不是本机加速证据。它们分别要求区分专家常驻成本与每token计算、共享执行同时保留状态分段、维护请求/取消/缓冲区生命周期。当前专家已在GPU常驻，直接照搬CPU/PCIe expert-cache路线不对应实测瓶颈；vLLM的mixed backbone、packed prefill、异步codec已经移植并验证过，不能重复计作未实现机会。

| 方向 | 当前状态 | 不能再据此提出的未经证实收益 |
|---|---|---|
| 原MoE布局、宽N、generic144 dense配置 | 已有失败或无净收益 | 再扫tile、只据低tensor活跃率承诺大幅提升 |
| expert2分区、共享metadata、resident完整链 | 精确gate通过但完整调用更慢 | 把较少global中间buffer自动等同于更快 |
| M128与PDL/metadata-ready | 最新局部约2.7–4.1%，未达continuation门 | 单GEMM或等待前移等同于服务显著提升 |
| cuBLASLt受限算法 | 本轮等速、无替换 | 原NO_CANDIDATE的A/A差量视为收益 |
| compact GDN、layout/conv guard、residual融合、reserved padding、双帧 | 已实现/验证的候选 | 重复提出成新架构 |
| codec cache/green资源/DP合并 | 已有固定工作与服务结果 | isolated codec下降直接转成服务同比增长 |
| FP32 prefix-state cache/完整prefix DAG | CPU方案，未实现；自然命中预算小 | 422个顺序重复当已就绪C64缓存命中 |
| **mixed MoE输入相同行合并/展开** | **本提案；未做hidden parity或GPU** | 只因原输入token相同就宣称hidden可以无损复用 |

NCU新的受限诊断由其他agent负责，本提案没有重跑该工作。

## 相同行必须满足什么条件

只接受本次已经到达的prefill请求，decode行永不合并。相同前缀需覆盖完整11列frame字节（semantic/text及声学code），相同checkpoint、模板/tokenizer与模型配置、相同绝对位置起点、相同参考/conditioning，且prefill初始conv/GDN为零、KV有效长度为零。不能用text token-ID或reference文件名代替完整条件。全前缀相同而后续target不同，在因果计算的数学定义下可形成相同prefix hidden；任意中间token相同则不够。

`mixed_graph`为新prefill独立slot reset，`mixed_model`每层分开因果attention/GDN后才进入 `moe_parts`。该MoE函数本身不读取请求history、位置或状态；它读输入行和固定权重，执行原共享专家、原top8路由和原FP8块缩放。sampling/history属于tower端，未被合并。此数学依赖审计不等于浮点位模式证明：packed位置、GEMM行数、attention/GDN分块选择都可能影响舍入。必须先在未改路径中检验每层的实际normalized输入字节相同。

首原型保持所有原padding行，避免把dummy-state改变混入同一变量。只合并真实prefix重复行；新容量的额外填充可复制保留行且不扩张真实专家集合。所有真实request state仍独立计算。跨批persistent cache、跨层cache、GDN跳算、teacher复用均不在首原型范围。

## CPU省行量：复用已有统计，不重新预处理语料

`derive.py`直接读取既有 `per-request.json` 和 `admission-proxy.json`，按原198组映射；196个mixed组的prompt计数和token总数逐一匹配当前raw timeline。它仍是假定FIFO corpus顺序的代理，因为服务trace没有request IDs；匹配token和不证明全部成员完全相同。

既有 `prefix-reuse-probe.json` 记录全1088请求共同前25行。本次在同组reference前缀去重之后，再合并每个不同reference的这25行，避免双算。

| mixed的候选合并 | 行数 |
|---|---:|
| 同组相同reference的完整因果prefix | 50935 |
| 不同reference之间额外共同25行 | 13650 |
| 合计真实prompt重复行 | 64585，即183228的35.25% |
| 相对所有decode+prompt+padding行222064 | 29.08% |
| 保留原padding，MoE容量按256向上取整 | 实际省38656行，17.41%，128/196组可用 |
| 同上，新增64行容量步长 | 实际省58560行，26.37%，181/196组可用 |

这不是prefix-state方案的C128截断：这里没有recurrent restart或teacher边界，因此可以考虑任意已证明相同的prefix长度。首原型可以只选最有利且固定的已记录group190（corpus1040起8条），原MoE1600行、代理重复661；256对齐可减少512行。若这个高机会组都没有净收益，则先停止而非立即扩展所有图bucket。

## 固定权重与token可变成本

去掉重复行后，真实active expert的并集不变：每个被删行仍有相同保留行，原padding也保留。因此每层参与专家的原FP8权重首次读取字节数并未减少。一个专家的FC1+FC2逻辑权重为 `2048×1024 + 512×2048 = 3145728 bytes`；全256专家约0.805GB/层。不能把减少35%行说成减少35%专家权重。

已有真实mixed D64/P1280的六个NCU样本，DRAM读量约1.011–1.017倍独立专家权重；原B64约1.0005–1.0014倍。范围只覆盖三个层/一个位置，不能推广成全40层当前服务的完整下界；但足以否定存在大量可以靠行去重自动删除的HBM重复权重读取这一假设。

能变小的是FP8输入准备、路由/expanded行、FC1/FC2的token算术和片上供数、activation、finalize，以及MoE内部BF16共享/router投影。首原型**不**减少QKV/GDN输入输出投影、attention/GDN核心、teacher或codec。

较早固定D64/P1280 profile每图FC1+FC2内核求和7.700ms、activation1.475ms、finalize0.504ms；它不是当前service kernel账本，且不能与重叠事件直接相加。仅把后两项按行数线性估计，256容量方案整轮服务只有约0.0569秒、64容量约0.0862秒的历史成本代理。要产生有价值收益，还必须看到专家GEMM token算术和共享投影随行数减少而下降；这是下一实验要证实的核心，不能预先承诺。

按每个实际mixed事件跨度乘相应省行比例，极乐观地连不变的状态/teacher/固定权重都线性缩小，256和64方案分别得到0.7963与1.2584秒；任意行理想合并为1.3981秒。它们是用于筛选的高估预算，不是严格硬件上界或服务预测。真正MoE-only还要扣两次gather/展开、图容量与原生kernel分派影响。现有证据不支持该方案达到整步20%或单独跨越6765；也不据此宣称原生目标不可能。

## 最短独立GPU可行性门（尚未执行）

1. 冻结当前source和一个真实固定mixed组，不改变任何服务配置。恢复同一decode/prefill初始state，在原路径hook每一层MoE输入；以完整prefix身份建立alias，再检查40层所有拟复用行的normalized hidden byte equality。包含有重复reference、不同reference共同25行、无可用重复、不同suffix长度；任一非精确行不允许被静默合并。
2. 输入证明后，首原型只调用 `compact = normalized.index_select(0, keep)`、原 `model.moe_parts(compact, layer, decode=True, row_slots=slots[keep])`，再按inverse map展开routed/shared/shared_gate三个结果，继续既有residual/RMSNorm。需要显式传row_slots；禁止可变model.current_slots。所有保留真实行仍完整top8与原128块FP8量化。记录原生GEMM tactic/dispatch，batchshape改变可能换算法，不能以函数名未变代替精度门。
3. 逐层比较实际输入FP8 bytes/scales、router logits、top8 IDs/权重以及展开后三个parts、全hidden与conv/GDN/KV状态、sampler/RNG和非参与slots。native路由物理permuted顺序可以变化，必须各用自身live metadata还原logical行；不能拿输出值排序。先eager，后同一capture的动态alias更新和slot reuse。所有失败保留，不用WER容忍错state或错映射。
4. 全精确后只做固定组完整mixed图AB/BA/AB：包含所有gather/expand、全部state操作、teacher和padding，记录额外graph存储。建议先以真实高机会组净省至少1ms/图作为继续到普通组的筛选门，而非仅看MoE子函数；门槛需由root在实验声明中确认。若最佳组无收益即停止该路线。再通过一般组后才谈full1088与WER/SIM；全部C仍需最终独立验收。

后续可扩展到逐行dense，但必须先证明MoE-only值得做；完整prefix-state/DAG及跨批缓存不随首原型自动获授权。无需重写Engine语言，也不把新的参数扫描包装成架构优化。

## 证据入口

- 当前服务：`current-pipeline-service-ledger.json`、`current-pipeline-timeline.json`。
- 前缀：`reference-prefix/{counts,admission-proxy,per-request}.json`、`reference-prefix/implementation-proposal.md`、`prefix-reuse-probe.json`。
- 历史kernel预算：`mixed-profile/table.txt`；旧mixed计数器为 `../drama3-mixed-20261016/ncu/mixed/`。
- 当前接口：`drama-3-single-b300-inference/src/drama3/{mixed_model,model,mixed_graph}.py`，实际阅读的hash见provenance。
- 三库指导：OmniRSI `src/omnirsi/data/repo_packs/strata/knowledge/common/llm/{architecture-search-space,controlled-execution-tuning}.json`；`vllm_omni/knowledge/cuda/audio/concurrency-scaling-state-and-overlap.json`；`sglang_omni/knowledge/common/omni/pipeline-state-and-transport.json`。

SAFE_TO_ARCHIVE。所有数字为CPU派生或明确标注的既有证据，不是新GPU结果；运行时未修改，原实验全部保留。
