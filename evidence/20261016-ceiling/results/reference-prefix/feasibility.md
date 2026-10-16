# Reference-prefix 状态复用：CPU 可行性与预算

2026-10-16；只读 runtime，未初始化 CUDA、未加载 GPU 模型、未改 runtime、未做 GPU 实验。最新服务源码 `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`。统计见 `counts.json`、`per-request.json`、`admission-proxy.json`，驱动及 hash 同目录保存。

**结论：真实 reference 前缀可复用，但不存在可直接打开的完整 prefix-state cache。高并发最大的重复来自同批请求，需要 prefix 去重后 state fork；这是一项新的续算接口改造，不是简单加一个字典。CPU 预算不足以承诺显著服务收益，本轮停在设计与预算。**

## 1. 精确输入前缀与合法复用

读取当前 Fish `generate.py:94–202` 的真实 `build_drama3_prompt`；CPU 驱动通过 AST 提取原函数与所需常量，使用 checkpoint 的真实 tokenizer/config，并逐条断言观察到的 prefix 张量等于完整 prompt 的前缀。没有复刻另一套字符串模板或将拼接文本重新tokenize。观察函数只截取原函数的 segment 列表用于计数，原 builder 本身不变。

单 reference 的顺序为：system头 → reference指令 → reference transcript → reference的全部10 codebooks → system结束 → user头 → **目标文本** → user结束 → assistant voice头。缓存边界P取目标文本之前，包含user头，不包含任何未来目标文本。缓存key应覆盖真实`[P,11]` token/code字节、checkpoint/tokenizer/template/runtime精度版本；不能只按音频路径或reference SHA推断。无history的本语料中相同audio确实对应相同prefix，通用服务仍必须检查实际prefix。

| CPU 精确统计 | 数值 |
|---|---:|
| 请求数 | 1088 |
| 唯一reference audio / 唯一prefix字节序列 | 666 / 666 |
| 同一audio对应多个prefix | 0 |
| 完整prompt总行数 | 185006 |
| 全部prefix总行数 | 163110（88.1647%） |
| prefix行数 min / median / p95 / max | 106 / 147 / 191 / 240 |
| target suffix行数 min / median / p95 / max | 11 / 20 / 25 / 37 |
| 冷缓存、完成即发布的顺序复用上限 | 422 hits，63681 backbone行（34.4210% prompt行） |

88.16%不能当可省行数：每个reference第一次到达必须计算。422次重复的**前次使用距离全为1**，这尤其不能当高并发已完成缓存命中。

## 2. 自然命中与同批去重必须分开

冷缓存按顺序逐请求完成后发布，任意足够小LRU均可得到422 hits；这是理想上限。简单按8请求一组、组完成后才发布的模拟只有53 hits / 8022行（4.3361%总prompt）；同组首见reference没有现成state。

新 `current-pipeline-timeline.json` 记录198个prefill/mixed到达组，共1088请求。按客户端FIFO corpus顺序将请求放入记录的组大小，196个mixed组的prompt token总数**全部吻合**。但timeline没有每组request ID，这仍是**FIFO代理**，不是实测cache命中日志：

| 冷缓存FIFO代理 | 请求/可省行数 |
|---|---:|
| 组开始前已有完成prefix | 82 / 12146行（6.5652%） |
| 同组另一个新请求的重复prefix | 340 / 51535行（27.8558%） |
| 两种机会合计上限 | 422 / 63681行（34.4210%） |

同批去重可以只观察已经到达的请求，按实际prefix相等分组，因而不需要任何未来目标预热。它需要“本批先算唯一prefix，再为每个请求分叉状态并算各自suffix”，不会被普通完成缓存自动获得。为了凑同prefix等待更多请求或重新排列语料，不在本提案范围。

现有合法64请求warmup单独估计可使理想顺序hit变成462；不把它算入冷缓存主预算，不增加warmup、不扫描整个reference库。所有正式比较必须继续采用冻结plan的 `max(8, concurrency), no reference-wide warmup`。跨重复/跨concurrency缓存是否保留也必须事先声明，不能拿上一轮已见整语料的暖缓存与baseline冷轮直接比较。

## 3. 必须保存的原生状态与 teacher label 边界

冻结 `model.py:217–228`：30个GDN层，每slot有BF16 conv `[8192,3]` 与 BF16 recurrent `[32,128,128]`。`:291–296`：10个backbone attention +13个tower层，各自K、V为BF16 `[position,2,256]`。

保守的prefix snapshot包含：

- 30层GDN在P行之后的状态，共30MiB；30层conv历史共1.40625MiB。
- 10层backbone KV的 `[0:P)`，每prefix行20480bytes。
- 13层teacher tower KV的 `[0:P-1)`，每prefix行26624bytes。
- prefix最后一行backbone hidden 4096bytes，以及边界P、最后原始frame等极小元数据。每请求RAS history/position应从自己的完整prompt初始化，不能缓存/复制全局sampling_step。

为什么tower少一行：`model.py:703–750` teacher labels用下一行，边界P−1的label为目标首行；保守实现须用cached hidden重算这一行，再执行suffix tower。不能把另一请求的目标相关label写入共享快照。

当前Drama3合法目标文本首行是非semantic文本/说话人token，10个code均0；EOS占位也非semantic且code为0。现有tower在`model.py:659–667`、`kernels.py:236–271`以semantic mask将embedding贡献置0，因此此特定输入契约下边界label数值不同理论上不改变tower计算，可以把tower缓存到P。**这需要一个明确native边界proof，并绑定text-only schema；不能泛化到history/audio semantic边界。** 当前预算取更保守的P−1版本。

快照必须独立于活跃slot；不能长期引用随后会被decode修改的conv/GDN/KV。发布需producer-ready事件，只有完整snapshot才可见；fork复制需在suffix读取前完成。淘汰或slot重用不能释放仍被CUDA graph/replay读取的快照。各branch的可写状态必须私有，只读prefix KV若采用共享页则另需page-lifetime管理。

## 4. 内存和复制成本

以上保守BF16 snapshot的平均payload为39,941,600bytes（约38.09MiB），范围约36.15–42.17MiB。全部666个约26.60GB（24.78GiB）；无需为了相邻复用常驻全部666个。示例LRU64约2.36GiB，LRU256约9.50GiB，未含allocator/graphs/索引等开销。

冷缓存全部666次存储+422次恢复的理想payload复制量为43.49GB（约40.50GiB）。单次mean snapshot以**假设**有效payload带宽1TB/s计约39.94µs，所有复制约43.49ms；若用内存总读写流量计则约86.98GB。此处只是`bytes/bandwidth`条件算术，没有测得带宽或服务时延。106个大状态张量分散copy的launch成本、与MoE/codec争带宽、CPU/offload与分配成本均未包含；不应写成“复制成本只有40µs”。

保留FLA内部未舍入的FP32 recurrent boundary会把GDN snapshot从30MiB增加到60MiB；BF16最终状态快照不等于原full-prefill内部每个chunk的FP32暂态。不能为了节省内存默默改变原生中间精度。

## 5. 当前路径为何不能直接 suffix 续算

1. `model.py:753–758`的`prefill_packed`及`mixed_graph.py:87–89`显式清零conv/recurrent；恢复快照后直接调用会被清掉。
2. `packed_prefill.py:112–123` conv对序列开头负索引补0，不读取恢复的三帧conv历史。续算必须读取prefix历史并正确更新末三帧，不能只跳过index_fill。
3. `gdn.py:146–163`底层FLA kernel支持initial_state/index并可原地写回，故有构建续算路径的基础；但当前layout/chunk从0开始，prefix/suffix拆分新增状态保存及BF16舍入边界。即使64对齐，也必须证明与原chunk内部状态/舍入相容，不能据数学递推就宣称bit exact。
4. `model.py:365–390` prefill attention虽然把K/V写入slot cache，attention本身只对当前局部q/k/v做causal self-attention，没有读取已有prefix KV。续算需要q=suffix、kv=prefix+suffix并使用正确absolute rotary position、causal offset/length。把position加P但仍用现有self-attention会直接漏掉prefix。
5. `packed_prefill.py:row_metadata`positions从0开始；mixed metadata也如此。续算必须各branch带独立prefix length，并保持prompt labels在branch内移位。
6. 首次miss目前直接完整prefill，没有P时刻完整状态导出。不能从最终状态倒推出prefix GDN。要么增加中间state tracking，要么拆prefix/suffix两段；两者都需专门实现/验证。

现有`engine.py:740–758`仅缓存reference**音频编码结果**，不是backbone/tower prefix state；预编码corpus已经绕过它。本轮搜索没有发现完整reference-prefix state复用实验。冻结plan没有禁止服务自然到达的相同前缀复用，但明确禁止reference-wide warmup；固定vLLM部署prefix cache=false应透明披露，不能修改其已冻结分数。

## 6. 最小可评估设计（尚未实现）

以当前已到达admission组为界，对真实prefix bytes去重。已有只读快照可直接fork；同批新prefix只算一次，每个请求正常算自己suffix，不改变输入、采样、chunk输出或EOS判定，不为凑组等待新请求。

最小结构是两段：先用唯一prefix运行无采样backbone/teacher prefill并导出快照；复制到每个branch slot；用新增suffix接口继续GDN/conv、prefix-aware attention和teacher tower；最后按既有顺序调用first-frame sampling，sampling_step只在原先的采样阶段递增。对已有active decode，仅在本轮对应的一段推进一次，不能prefix阶段与suffix阶段各推进一次。

这种结构**会额外遍历整40层权重**：原先一次mixed projection/MoE处理prompt+active rows，分开prefix和小suffix后可能增加权重扫描、kernel启动，并使suffix MoE/GEMM batch太小。即使减少34.42%prompt行，额外两段与状态复制也可能完全抵消；同批state fork不是免费的branch引用。

如果后续决定验证，最小门槛应是：一个真实相同prefix、不同target二branch，从独立full-prefill控制对照prefix state、suffix hidden/全conv/GDN/KV、闲置slot、首帧codes/EOS；分别覆盖边界内/跨64chunk、带/不带现存快照和slot重用。精确失败必须保存并定位舍入来源；如果仍属原生计算变化，单独声明native数值诊断，再做全1088 WER/SIM。最终promotion仍只能用同源完整E2E和冻结三次所有并发验收。

## 7. 可恢复服务预算

最新e7cda1e profile的196个`mixed_backbone` span累计4.946353s，含teacher prefill与旧请求backbone；recorded prompt tokens183228/capacity209664，平均5.505新请求+56.06旧decode。不能把4.946s全当独立prefix成本。

把冷缓存+同批去重上限34.4210%粗略乘整个mixed span，得到**约1.703s线性行成本预算**，占本轮22.895s请求窗口约7.44%。这已经给cache过多利益：将旧decode也按prompt比例削减，忽略MoE非线性和额外权重遍历。它是审计用的乐观粗估，不是严格时间上界、硬件roofline或实测收益。只考虑82次已有完成快照的FIFO代理，类似粗估仅约0.325s，且miss时导出/拆分仍有成本。

因此这不是单独弥补当前4726.77→6765.41差距的可信路线。它在频繁复用speaker/reference的真实产品流量上可能值得做，但本语料C64下“加prefix cache即可大幅提升”的说法不成立。本轮停在CPU可行性，不直接投入复杂runtime改造或GPU实验。
