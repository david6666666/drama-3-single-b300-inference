# 原生参考前缀复用：可执行边界与状态接口提案

本提案仅做 CPU 源码审计，未修改 runtime、未启动 GPU，也没有重跑 tokenizer、语料前缀识别或已有 cache 模拟。原统计继续以 `counts.json`、`admission-proxy.json` 为准；新的 `alignment-boundary-proof.json` 只将已记录请求和组号映射到一个更保守的 128 行边界。当前源码接口可实现续算，但不能直接恢复 BF16 最终状态然后调用现有 prefill，也不能把同组重复当作已经就绪的 cache hit。

最小路线是“完整 miss 原样执行并导出内部边界；已完成 hit 只执行 suffix”。它不需要为了导出快照把 miss 拆成两遍完整网络；但在本语料高并发代理下，收益预算只有约 0.243 秒，工程成本较大。若希望利用主要的同组重复，应采用后述逐层 owner/fork，避免简单 prefix/suffix 两遍网络额外扫描全部 MoE 权重。后者是一项新结构，尚无精确性或速度证据。

## 1. 保守边界：严格早于 target，且对齐原始 chunk

令 P 为已证明完全相同、目标文本之前的前缀长度，选择 `C = floor((P - 1) / 128) * 128`，C=0 则走原路径。本语料 P=106…240，所以非零边界只有 C=128。

- 当前 GDN 的实际 chunk 是 **64 行**（`gdn.py` 的 BT=64、chunk_local_cumsum=64）；128 是本提案的保守倍数，不是把原核误认为 BT=128。续算第 j 行属于原第 `C/64 + floor(j/64)` 个 chunk，不新增半块分割。
- `C<P` 保证 teacher tower 缓存位置 C−1 的下一行 label `frames[C]` 仍属于共享前缀。因此可以安全缓存 tower KV `[0:C)`，无需依赖“目标首 token 非 semantic”的特定 schema，也无需缓存最后 hidden 或重算 teacher 边界行。P恰好128时故意不缓存，不能静默把目标首行纳入 key。
- key 至少包含真实 prompt frame bytes `[0:C+1)`、checkpoint、tokenizer/template、native kernel/精度协议版本。最小实现直接沿用已有完整 prefix hash 更保守；应校验字节而不只用音频路径。采样步、请求 history、最大长度、取消状态不属于快照。
- 3,645 个纯整数边界组合覆盖既有 P范围×suffix范围11…37，验证 teacher lookahead、64chunk对应及 causal offset。它不是浮点精度证明。

## 2. 快照内容与 miss 导出

每个 C=128 条目保存：30层 **FP32** recurrent `[32,128,128]`；30层 BF16 conv 历史 `[8192,3]`；10个 backbone attention 和13个 teacher tower 的 K/V `[0:C,2,256]` BF16；边界、版本、ready event及refcount。payload约 **70,418,432 bytes / 67.15625 MiB**，不含索引/allocator/graph额外容量。LRU16单payload约1.049 GiB；不需要保存全部666个reference。

miss继续原 full/mixed prefill，不改变输入和采样顺序：

1. 各 GDN 层在原生 `chunk_delta_h.py` 的 `TRACK_STATE` 分支设置 `track_chunk_idx=C/64`。该分支在循环第 i_t 块计算前保存当前 FP32 `b_h`，因此导出的是消费 `[0:C)` 后、尚未额外 BF16 舍入的状态。当前 wrapper 设置 `TRACK_STATE=False`，必须新增显式可选导出参数。缓存最终 `model.states` 或 BF16 `h` 中间张量均不等价。
2. 每层 conv 快照取其 packed input projection 的 `[C−3:C)` 三行，转成现有 state layout；不能复制 full prompt 完成后的 conv state。保留原 BF16 投影输出位模式。
3. KV 导出 `[0:C)`。可以每层写入私有 snapshot，所有层及 teacher 完成后统一 record ready event，再发布。正在 decode 的原 slot 后续可继续被修改，snapshot不得引用该slot的可写存储。
4. 只对本次已经到达且选择建立条目的 miss 导出；无预扫描、reference-wide warmup、未来 target 预计算。cache淘汰必须等所有已捕获读取结束；同批 owner尚未完成的条目只能是pending，不能对外标ready。

新增 TRACK_STATE stores 仍可能影响编译和执行，因此必须对 miss 分支做“额外导出不改变原全输出/全state”的精确验证。不能单凭它只增加存储就声明通过。

## 3. 已完成 hit 的最小续算接口

建议独立 `ContinuationLayout(boundaries, sequence_slots, base_positions, snapshot_ids, initial_h32, kv_lengths)`；原 `PackedLayout` / 默认路径保持原行为。图固定地址，host仅更新metadata；不得在capture中做cache查找/分配。

- **reset与位置**：`mixed_graph.py:forward` 和 `model.prefill_packed` 当前清空所有prefill状态。续算图对hit跳过清零并恢复conv；miss仍清零。positions=`base_positions+local_row`，各branch/active decode/private padding slot必须不重叠。DeviceRequests history/位置仍按各自完整prompt初始化。
- **conv**：当前 `packed_prefill.convolution` 对负source补0，必须改为读取snapshot三帧历史。suffix最后三帧的state更新也要覆盖一般 length0/1/2，不能留下旧history；本语料虽suffix>11，接口不能默默依赖它。
- **GDN**：必须以FP32 boundary作为suffix第一块initial state。当前原核把 `initial_state` 同时当最终state输出，直接改为FP32会改变服务state dtype。最小可行方式是私有FP32 temporary作为原核in/out，保留原chunk运算和BF16 `h`/updated boundaries；完成后显式按原最终存储边界转换到BF16 request state。更干净但需独立源码审计的是显式拆开FP32 h0与BF16 ht指针。两种方式均需逐层精确gate，不能假定加载时的零加法、FTZ或编译选择不影响位模式。
- **attention**：当前 prefill attention只读本次q/k/v，恢复KV后直接调用仍会漏prefix。需要q=suffix、kv=`cached_prefix + new_suffix`，分开Q/KV boundaries，绝对RoPE position和causal条件 `key_index <= C + suffix_query_index`。优先保持原ATen FlashAttention路径并显式验证其 unequal-Q/KV causal语义和kernel选择。128对齐只能减少tile边界变化，不能保证与原整段kernel bit exact；本机独立flash_attn/cute bottom-right源码只能说明另一实现有该语义，不能替ATen作证明。
- **teacher**：KV `[0:C)`已包含正确lookahead；suffix teacher labels仍按各自完整prompt移一行，最后保持原EOS占位/first-frame覆盖规则。不执行任何新sampling，`sampling_step` 仍只在既有first-frame阶段推进。
- **调度**：只处理原 admission 已选出的请求，不为了cache凑组等待。旧active decode每轮只推进一次；继续保留“新请求first-frame采样，再旧decode tower”的现有调用顺序。取消和slot reuse必须释放snapshot引用并保留GPU completion event，不回收仍在读取的条目。

## 4. 同组重复：避免两遍40层权重的逐层 fork

已有代理表明主要机会是同组340个重复，普通 completed-cache 最多82个自然命中。简单“所有prefix先跑完整模型，再所有suffix跑完整模型”将额外扫描全部40层dense/MoE权重；高并发原MoE已接近一次权重读取，这个代价不能忽略。

可执行但更复杂的结构是：每组一个owner保留完整prompt，其他相同prefix的branch只保留suffix行；所有active decode、owner完整行、branch suffix行**每层只做一次共同输入投影和一次完整top8 MoE**。

在该层状态算子内部：先对owner运行原GDN并导出C处FP32边界，再对branch suffix运行原GDN continuation；intra计算可共同打包，h更新分两阶段依赖。branch conv负索引可直接读取owner该层projected `[C−3:C)`；attention为branch拼入owner该层已生成的prefix K/V。teacher各层也以owner prefix KV提供causal历史，C<P保证label无target污染。完成该层后，各branch继续持有自己的可写state；下一层再重复该结构。缓存命中可用已有snapshot替代owner。

这样避免额外完整MoE权重遍历，但增加逐层状态fork、额外GDN h launch、KV组装和metadata，以及不同packed row-count下的GEMM/attention算法选择。不能承诺数值逐位相同或计算行减少就等比例加速。应先单个真实双branch验证该层DAG，再考虑全40层；不要先做通用paged-KV/radix-cache框架。

## 5. 绝对时间预算与6765差距

不重跑既有模拟：把 `per-request.json` 的重复条目直接映射到 `admission-proxy.json` 已记录组号，按上述C边界派生：

| 机会 | 原proxy重复数 | C=128可用数 | 可省backbone行 |
|---|---:|---:|---:|
| 已完成快照 | 82 | 71 | 9,088 |
| 同组owner/fork | 340 | 290 | 37,120 |
| 合计 | 422 | 361 | 46,208 |

它们仍是FIFO代理，不是实测cache命中。合计约总prompt行的24.9765%；原任意P理想复用为34.4210%。用既有mixed总span4.946353秒线性分摊：最小 completed-cache 方案约 **0.243秒**；包含同组fork约 **1.235秒**；任意P理想方案原预算 **1.703秒**。这些还没有扣除FP32 snapshot/恢复、带宽竞争、额外h阶段、KV组装、graph容量取整，也错误地让旧decode工作一起随prompt行减少，因此是偏乐观的审计预算，既非实测收益也非严格时间上界。

22.895秒profile窗口中，1.235秒约5.40%，1.703秒约7.44%。最佳未profile吞吐4726.77到6765.41需要约30.13% elapsed reduction。仅作归一化示例，把这些比例套到最佳结果分别约4996和5107 fps，仍距目标约35.4%和32.5%的额外吞吐提升；**这不是把不同运行组合成测量结果**。因此前缀复用没有现有证据能单独填补缺口；自然命中最小实现的0.243秒预算尤其不足以支持立即大规模runtime重构。

## 6. 下一步仅在预算认可后做的门槛

第一门槛应是一个原生真实相同prefix双target：owner保持整段控制，C128snapshot+suffix分支逐层对照FP32 tracker、BF16conv/GDN/KV、全部有效hidden、first-frame codes、RNG及非参与slot；覆盖P128（fallback）、P129、长prefix、slot reuse和cancel。先保存全部exact失败并定位，不能放宽旧门槛覆盖失败。若只满足同原生数学而finite-precision路径变化，明确作为独立候选，完整1088 WER/SIM仍不可省略。

第二门槛才是固定同一 admission组、相同生成码/状态和graph容量的成本比较，计入snapshot、fork、metadata、所有kernel、旧decode一次推进及teacher/first-frame边界；与decode/codec竞争另作完整服务验收。本提案没有启动这些GPU实验。已有统计/输入/报告保持原样，M128全部冻结结果未动。
