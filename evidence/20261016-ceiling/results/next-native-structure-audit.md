# 下一轮原生结构优化审计

本次只读源码、复算已有 profile，并运行 CPU prompt census；没有模型/GPU试验或运行时修改。固定计算约束仍是原生 top-8、原始128-block FP8专家权重/scale、BF16 dense与原生GDN运算。质量界限不变。C64要求6765.407 frames/s；尚未达到，更没有因局部收益预算较小而停止优化的结论。

审阅基准源码为 `source-codec-sm-69d636e`（69d636ef11875add6e4edf7c9bb028b9574143a3）。当前分支已有463d837 reserved-padding router及a251689跨帧pipeline，均不重复建议为新发现。Green SM、coalescing的服务/质量由root收尾；本文不自行晋级这些候选。

## 判断与下一步

**最值得继续投入大块原生计算的是两条不同风险的路线：按专家驻留的 FC1→原生SwiGLU/FP8量化→FC2 管线，以及逐层批内共享前缀DAG。** 前者覆盖纯decode中约42%的两次专家GEMM及约4%的activation（旧profile内核求和），后者能减少真实语料批内约23%的prompt行，但只影响prefill侧且实现复杂。它们都需要先做隔离结构原型，不能从份额直接宣称同等提速。

较短验证路径可并行选择 **GDN recurrent H与chunk O的生产者/消费者融合**，不再重复compact无效tile或mixed cat消除。`do_finalize=False`归并融合可以做，但预算与风险都应如实列出：它是覆盖每个MoE层的小优化，不能充当大幅目标的主路线。Token预算admission应先补实测成员与cost元数据；仅换一个固定batch阈值不是新结构证据。

## 已测预算与不能推出的结论

| 已有证据 | 实际观察 | 对下一步的约束 |
|---|---|---|
| `../drama3-mixed-20261016/ncu/README.md`，B64层0/19/39 | 六个FC1/FC2读量约为活跃专家权重的1.00052–1.00138倍；DRAM peak56.36–75.54%，Tensor active3.53–4.48% | 没有大规模重复读专家权重证据；不能把低tensor百分比说成20倍可用算力 |
| 同目录`ncu/mixed/README.md`，D64/P1280 | 六个GEMM读/独立权重1.0110–1.0170，DRAM peak49.36–60.90%，tensor17.93–22.66% | mixed仍需读近一份所有参与专家；减少重复token主要省activation/算量，不会等比减少专家权重 |
| `../drama3-scale150-20261015/native-bf16-projections-b64/table.txt` | 3次replay内核总和28.224ms；每次FC1+FC2=3.957ms，activation=.356ms，finalize=.323ms | 这是旧纯decode profile的内核求和，不能替代新图墙钟或当前服务基线 |
| `mixed-profile/{metadata.json,table.txt}` | 实际D58/P6，捕获D64/P1280，隔离28.322ms；3次profile内核求和每次29.501ms。FC1+FC2=7.700ms，activation1.475ms；H .967ms、O .978ms、KKT .636ms、W/U重算 .535ms；finalize .504ms | 内核求和含重叠，不与service event span相加；已有compact/layout/guard/residual改变了部分小项 |
| `rich-timeline-service-ledger.json` | 1088请求、23.23366s；pure decode事件16.013s；mixed backbone含teacher4.900s；first .234s、old tower .289s；持续期active约60.85 | 主战场仍是实际执行的decode与mixed。全程低平均active不能冒充稳态空槽；不能删除尾部 |

孤立B64的8.275ms→7734.139 frames/s仅是实测参考，不是硬件上限或服务下限。目标/参考=87.47%；较新的未profile服务对照4412.55仍需约53.32%吞吐增长。新结构可以改变这份参考；本文不将目标判为不可达。

## 1. 原生按专家驻留的FC1→activation→FC2

精度契约更正：安装版FI DeepSeek路径的FC1输出是E4M3＋FP32 scale，并非BF16；activation再量化E4M3，FC2才输出BF16。先前版本已保存在`next-native-structure-audit.before-fc1-contract-correction.md`，不可按旧BF16边界实现。最小原型与分组复用vendor GEMM的较低成本路线见`expert-pipeline-first-layer-plan.md`，该计划优先做2×128分组可行性，再考虑自写tensor-core主体。

当前 `model.py:513–573` 每层调用同一原生FI block-scale MoE；profile显示路由后分别执行两次GEMM和独立`activationDeepSeekKernel`。既有majorK/blocked及tile/autotune实验已经做过；这里的变化是**跨两个GEMM的中间数据生命周期**，不是重新扫tile。

最小原型仅替换一个真实B64层的专家主体：沿用FI原生top8及permuted row metadata，每个专家/专家tile在持久CTA/CTA cluster中完成FC1，将原生FC1 E4M3字节/FP32 scale、反量化后FP32 SwiGLU、第二次128元素FP8 quant/scale保留在共享内存或明确受控中间区，再执行FC2，最后仍走原生finalize。所有真实专家继续计算，原权重字节/scale不变。输出tile若需要跨CTA访问同一activation，需要明确cluster同步或生产者/消费者barrier；不能假设普通CTA共享内存跨block可见。

可能收益来自少一次全局activation往返、少launch/跨kernel依赖，以及小M expert中两个阶段的供数协调。**3MiB左右/专家的FC1+FC2权重仍必须读，不能把整个约4ms GEMM项当成可省成本。** 当前读量已接近独立权重一次，所以若只是把数据搬运换成更低占用的大kernel，可能更慢。混合1344行要单独判断，先限制原型B64，已有原生路径fallback。

验证门：先从已存真实层0/19/39输入和FI路由导出参考中间结果；逐项比较输入FP8 bytes/scale、FC1 E4M3/FP32 scale边界、SwiGLU后E4M3/FP32 scale、FC2 BF16和finalize。相同top8、dtype但改变FMA归约次序仍可能非bitexact，必须量化误差并保留失败，不能默称“无损”。然后全40层B64、partial58/63和混合回放，所有真实/非参与状态、RNG、EOS/maxframes/cancel/slot-generation；最后同源服务与完整质量。第一门先看相同固定状态真实GEMM主体是否有实质净收益，而不是盲做全框架重写。

## 2. 批内逐层前缀DAG：独立复算后有机会，但不是27.9%服务收益

新增复算入口 `prefix_intrabatch_probe.py`，结果 `prefix-intrabatch-probe.json`，运行log同名。1088原始prompt总185006行，CUDA未初始化。key匹配全部11列并多匹配一行teacher lookahead，只计64对齐前缀，所有请求原顺序保留。

| 合成FIFO组 | 批内可共享请求 | 重复prompt行 | 行比例 | 256容量总和：原→unique |
|---|---:|---:|---:|---:|
| batch6，offset0 | 359 | 44224 | 23.904% | 206336→163840，−20.60% |
| batch6，6种offset范围 | 343–361 | 41792–44224 | 22.590–23.904% | 容量下降19.21–20.60% |
| batch8，8种offset范围 | 362–378 | 44160–46336 | 23.869–25.046% | 容量下降22.26–23.59% |

batch6 offset0的182组中181组有共享，最大3个分支；共享长度为64×45、128×296、192×18。**真实prompt数据不等于真实admission成员**：现有rich trace只有prompt_count/tokens，没有请求ID，因此这些仍是synthetic分组。原先422个顺序ready命中/27.88%是跨批即时可用假设；ready-only batch6只有3.98%。两者不能互换。

逐层DAG能避免“先跑40层prefix，再跑40层suffix”的双权重pass：每层所有unique node输入已由上一层完成，norm/QKV/GDN projection先对unique行一次；该层prefix recurrence完成后再分叉suffix recurrence；全部输出行合成一次out projection及一次MoE，进入下一层。Teacher tower同样按层做，输入标签key必须含lookahead，不能只匹配文字/voice reference。

有三项必须付出的依赖成本：

1. GDN prefix→suffix需要保留原生FP32内部accumulator，不能拿现有BF16最终state做分叉。`sglang/.../fla/chunk_delta_h.py:167–207`保存BF16 chunk h同时提供FP32 TRACK_STATE；`235–328`继续更新FP32状态。对于一个共享prefix及其所有children，要让**第一个owner的suffix也从同一FP32检查点继续**，最终才写BF16请求state。每层最多3个branch的FP32 state fan-out、至少一次prefix/suffix阶段依赖，以及chunk pad都要计时。仅每层临时检查点不必长期缓存40层全状态，但图pool生命周期必须覆盖children。
2. conv保存prefix最后3个原生BF16投影输入，suffix前3tap读取独立immutable history。输出最终state和输入history不能别名，否则多CTA可能读到被提前覆盖的末状态。
3. suffix attention需prefix+自身suffix KV，不能用现有只看当前packed suffix的方形causal attention。非方causal对齐、绝对RoPE位置、所有10个backbone attention和13个teacher层都要验证。先gather KV证明语义，再决定共享pages；page/position ownership和padding分支不可混写。

乐观示意：把旧4.900s mixed全部视作按prompt行线性缩放，23.904%只相当于1.171s/23.234s=5.04%时间；这既不是严格上限，也不是预测，因fixed MoE权重、decode prefix与分支代价并不随prompt行同比下降。按当前NCU，独立专家的权重流量大多仍保留。因此该方向可能覆盖比finalize融合更大的部分，但**不支持单靠它达到1.5×**。

最短可验证路径：先记录实际admission request IDs→CPU建立unique64/128 prefix DAG与节点/padding账本→固定投影输入证明GDN FP32branch与conv状态精确→一层含原生attention的unique-DAG与重复行参考比较→40层+teacher完整状态→动态服务质量。不要先建大LRU，避免将3.98%ready-only误当23.90%批内共享。

## 3. GDN H/O生产者消费者融合

`gdn.py:125–207`当前依次norm/cumsum、KKT+W/U、H recurrent扫描、O chunk输出。compact只跳空chunk；mixed-layout只消除cat/zcopy。这些没有消除H写BF16 chunk状态和updated V、O再读它们的往返。

候选保留原生KKT/W/U，在同一seq/head/Vtile程序的每个64-token chunk中，按原H顺序得到updated V后立即计算对应O；然后继续下一chunk。必须保留：chunk开始的h **先cast BF16**供O使用；updated V在decay/state-update前 **先cast BF16**供O使用；H循环的FP32accumulator和两个64维dot顺序不变；O原BK128 dot、exp/scale与最后BF16 store顺序不变。不能直接用更高精度寄存器替代已有BF16中间边界。

好处是去掉巨大的h workspace读写及updated V往返/一轮launch；风险是O原来按chunk并行，融合后串行进H loop、寄存器压力上升，可能得不偿失。原H+O共约1.945ms/旧mixed28.322ms只是被覆盖的工作，不是可删除时长。先验证span128/256、真实长短不齐、空padding、初始state非零、finalchunk非64倍数；记录H/O完整输出及所有slot state。若确切运算顺序实现复杂，可先只融合packed conv→原BF16silu→q/k L2norm的生产边界，保持统计中的主预算与次预算分开。

## 4. FI finalize＋原生combine/residual/RMSNorm

安装的 `flashinfer/fused_moe/core.py:5656`确有`do_finalize=False`，返回`[gemm2_output,expert_weights,expanded_idx_to_permuted_idx]`。`core.py:3219`明确FC2输出布局由运行时专家分布决定；`shared/inputs.py:133–156` fake结果首维也是动态的。**不能把它reshape为[M,8,H]，也不能假定graph capture时分布不变。** 必须使用真实expanded→permuted映射、跳过−1、持有返回workspace直至consumer结束，并验证每次graph replay换route后的稳定地址与边界。

原native finalize在`.../trtllm_fused_moe_dev_kernel.cu:842–948`按top-k顺序用float累加后cast BF16。新consumer需先形成完全同样的BF16 routed值，再执行现有`kernels.py:449–500`的`fma.rn.bf16` shared combine、BF16 residual舍入和原RMSNorm。直接把FP32专家和shared残差合并会改变现有数值语义。小batch≤8继续原生small_moe fallback，不要为这条优化扩张验证面。

已启用FUSED_MOE_RESIDUAL1已经去掉独立combine+add_norm之间存取。新贡献仅finalize与现有融合consumer之间的一次routed write/read及launch，旧pure约.323ms、mixed约.504ms项中仍有必要8-way weighted reduction，不可能全删。可作为低层补充，但不是主要结构路线。门：三真实层的do_finalizeFalse+原native还原→所有40层单/多步与routing变化回放→full服务质量，原API路径fallback默认保留。

## 5. token预算admission：先度量，不把改阈值称为chunked prefill

`engine.py:423–470`按空槽/最多8请求取waiting，active<32或waiting≥6或100ms等待触发；prepare_prompt后才知道总tokens，256向上取整，mixed图只捕获≤2048容量。`rich`的197次mixed：183228有效prompt tokens/208384容量，padding13.73%；主要P1024/P1280。这是可以改善的调度面，但没有证据证明剩余问题只是固定batch6。

具体最小结构：CPU提前准备有限FIFO waiting的prompt/length，选择满足既有graph容量与实测decode阻塞预算的**整请求FIFO前缀**，保留age/deadline与低active优先补位；不得无限按短文本跳队。记录每次eligible/selected/requestID/token_sum/capacity/span/old_active/排队等待与实际mixed duration。离线对照“少量大prompt不阻塞decode”和“更多小prompt摊薄权重pass”的冲突，再预声明一个策略做单变量试验。旧请求24–28ms mixed暂停不是全可删工作；拆更多批会重复读40层专家权重。

现有MixedGraph.forward逐层reset所有新prefill状态，positions从0开始，prefill attention只看当前输入。真正分块prefill需要保存中间每层GDN/conv/KV、继续绝对位置、处理teacher lookahead，并延迟finish_prefill直到末chunk。只按token截断当前prompt会破坏模型，不能把scheduler改动单独包装成chunked支持。批内DAG continuation可共享这些基础设施，但仍要分别验证。

## 实施次序与验收

先为两条大结构完成**真实输入的一层隔离原型**：MoE持久主体、GDN/前缀branch。选择是否进入完整框架依据本地正确性与未profile实际净时长，而不是理论份额。与此同时finalize融合可作为小且独立的补充；token调度先补成员/cost账本。已测布局、tile、priority、SM reserve不再盲重复；必要的新shape只因新结构改变才验证。

所有路径保持默认关闭与完整fallback；过程结果分别标注bitexact、原生但有舍入差异、完整WER/SIM及吞吐，失败证据不覆盖。服务依次要求同冻结source/hardware/env、全1088输出与尾部、完整质量；想声称A/B提升再做至少3次重复，并继续覆盖所有并发。局部1–5%不是最终目标，多个方案也不能将预算简单相加：共享同一访存/调度成本会互相重叠。最终目标继续保持INCONCLUSIVE，接着用实测反馈选择下一结构。


## 后续单层可行性结果

原生2×128专家分组已完成：`expert-partition/summary.md`。135项FP8中间值/FC2/finalizer/graph ownership检查精确，但三个真实层的完整分组路径比native慢17.59%–24.68%，未达到5%局部收益门，不晋级服务。并行比串行分组快6.15%–10.50%不足以证明相对native获益。额外q/scale/logits拷贝与重复metadata构建尚无组件profile，不能将全部回退归因于某项。下一步仅CPU审查共享一次原生routing/metadata的launcher，原用户目标不变。
