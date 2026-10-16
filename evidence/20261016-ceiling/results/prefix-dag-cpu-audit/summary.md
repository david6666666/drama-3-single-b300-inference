# 从 MoE 行复用推广到完整 prefill：CPU DAG 审计

**同组完整前缀 DAG 在原生因果计算上可实现，但当前 MoE exact 结果不能证明它的 GDN、attention、teacher 状态正确，也没有足以单独弥补6765目标缺口的预算。** 推荐先完成已有MoE复用的完整mixed筛查；本报告没有启动完整prefill改造、GPU验证或新的cache预热。

## 与已有结果的边界

`native-row-reuse-exact-v2/capture_inputs.py:35–74` 的alias按真实11列因果frame bytes、绝对position和零初始状态建立，覆盖同reference完整前缀以及跨reference共同25行。三个已审实际输入case为proxy190、其反序、proxy192；40层MoE输入的相等与完整原MoE compact/expand exact gate已保存。它证明的是这些实际输入上的无状态MoE复用。完整mixed状态验证属于另一个实验，本报告不预判其结果。

同组没有“已完成缓存”前置要求：owner与branch都已经在当前正常admission里。71个ready命中仅属于原C128 completed-cache代理，不是所有同组重复的限制；反过来，同组owner尚未执行时也不能算ready cache hit。

## 可以保持原生计算的逐层图

一个最小结构从**C128**开始，owner保留完整prompt，branch只执行[C:end)，旧decode正常保留。每层输入norm/packed projection仍将所有unique行与旧decode共同处理一次，层末完整top8 MoE也共同处理一次；不要先跑完整prefix网络、再跑完整suffix网络，否则会重复40层权重扫描。

- **30个GDN层**：对owner执行原chunk流程，并在C/64处导出未额外舍入的FP32 `b_h`。branch suffix的intra运算可以先准备，状态h推进必须等owner checkpoint；以该FP32值继续原chunk recurrence，再按原最终边界写回branch BF16 state。下一层之前branch已有独立可写状态。不能只复制owner跑完全部prompt后的最终state，也不能复制BF16 chunk `h`当FP32carry。
- **conv**：branch的t<3输入从owner本层packed projection的[C−3:C)读取；其最终history写branch私有slot。读历史与写最终state不得共用会被并行CTA提前覆盖的存储。
- **10个backbone attention层**：owner prefix Q/K/V只算一次；branch query需要看owner prefix+branch suffix KV。当前 `mixed_model.py:81–92` 的ATen attention使用相同Q/KV boundaries，只看当前sequence，因此必须新增分开的Q/KV长度、绝对RoPE position、非方形causal offset，或者做明确KV组装。仅恢复KV pool后照调当前prefill会漏历史。
- **13层teacher**：同样逐层共享prefix KV；key要覆盖下一行label，即共享[0:C)须匹配frames[0:C+1)。C<P保守保证该条件。teacher没有新sampling，first-frame和旧decode tower仍按原调用次序，不能改变sampling_step/RNG/EOS契约。
- **所有slot和生命周期**：owner原slot、branch写slot、旧decode与padding互斥；owner checkpoint不可被branch原地更新，branch完成才释放当前层scratch；图replay前更新全部关联metadata，取消/复用只能在其消费者结束后执行。无需保留跨请求cache也能做同组fork，但不能把可写owner slot地址当长期cache。

源码约束：`gdn.py:130–180` 原h kernel固定BT64、TRACK_STATE=False；installed `sglang/.../chunk_delta_h.py:188–207` 的TRACK_STATE只在chunk开始导出FP32，`130–131` h0/ht共用initial_state指针，`331–348` 最终按目标dtype存储。最小实现需要显式immutable h0与branch ht或私有FP32续算scratch；若直接传同一owner state给多个branch，原INPLACE_UPDATE会造成写入竞争。`mixed_graph.py:83–85` 当前会清零所有prefill state，续算/导出路径也必须区分。

对齐并不保证bit exact：改变packed GEMM/FlashAttention形状、kernel选择、h0加载零加、FTZ等都需验证。原模型数学不变而重排造成数值差异时，必须保存原exact FAIL并定位后另开数值诊断，不能掩盖状态、路由或边界错误。

## 为什么共同25行不是现成的完整state快照

当前alias包括所有reference共同25行，因此case190有661alias而C128 fork只省512行；case192有531alias而C128 fork只省384行。**25小于原GDN64chunk**，现成TRACK_STATE没有t25检查点。把一个64行chunk拆成25+39会改变原intra/矩阵递推的分组与有限精度边界；这是同数学的新数值路径，不能继承MoE的bit exact证据。另一选择是继续完整GDN，只共享这些行的无状态projection/MoE，节省范围也相应缩小。

teacher还带一行lookahead。只知道共同25个输入frame时，保守复用24个teacher位置；位置24的label来自第25行，不同reference可能不同。现有非semantic mask可能进一步消除某些label差异，但没有额外原生证明就不使用这一特例。

## CPU行数与时间预算

新`analyze.py`只读取既有1088条元数据与198个admission组大小，未重建prompt或重跑模型。group190/192推导分别恰好661/531，与真实case捕获一致。完整198组仍是FIFO代理：196组prompt token总数全吻合，但旧timeline没有每组request ID，不能冒充实测全服务命中率。

| 完整prefill路线 | 可省行数 | 占185006 prompt行 | 按4.946353s mixed成本线性估计 |
|---|---:|---:|---:|
| C128同组fork | 37120 | 20.06% | 0.992447s |
| C128同组fork + 71 ready | 46208 | 24.98% | 1.235425s |
| C64同组fork，仍需attention数值检查 | 41472 | 22.42% | 1.108803s |
| 任意P + common25 的同组backbone理想DAG | 65285 | 35.29% | 1.745471s |
| 同上、teacher保守lookahead | 64395 | 34.81% | 1.721676s |

最后两行是不同model部分的行预算，不能相加。它们需要解决partial-chunk GDN与attention新接口，不是已完成实现。所有时间列都过于乐观：把mixed里的旧decode、固定权重成本也按prompt行同比削减，忽略fork、额外h阶段、KV组装、图容量取整及codec竞争；不是测量、严格上界或roofline。当前MoE复用若获得收益，也已包含在完整DAG预算内，不能再次相加。

同一profile的mixed+teacher总4.946353s，另2次纯prefill合计0.047793s，全部相关model阶段合计约 **4.994146s**。该次benchmark达到6765所需elapsed缩减为 **7.353450s**（ledger使用benchmark makespan；请求窗口22.895377s）。即使把这些阶段全部删除，这个算术账仍不够；实际还必须保留旧decode与unique prompt。这不是排除解除竞争带来的间接收益，但没有数据支持把间接项凭空补足。因此完整prefix DAG可作为更广泛speaker/reference服务优化，却不能被称为本轮目标的充分路径。

## 资源与最短验证面

C128完整cache payload为70,418,432B：30层FP32carry共62,914,560B、conv1,474,560B、23层KV6,029,312B。**同组逐层fork无需同时缓存全部30层carry**：一个owner每层仅2MiB，8owner+8branch临时FP32为32MiB，可跨层复用；各请求已需的最终KV/state仍保留私有。累计导出/恢复字节与额外launch不能省略，但这不是“容量太大必然不可行”的理由。

最短正确性门槛应是一层真实owner/branch C128：固定原q/k/v/decay/beta，导出+续算对照原whole sequence的所有hidden、FP32checkpoint、最终BF16state、conv和KV；随后才是30GDN+10attention+13teacher全图。需包括反序/换owner、动态group、slot复用、P128 fallback、P129、空/短suffix、teacher边界、旧decode只推进一次、padding和未参与slot。只有同源完整mixed计时含所有fork/metadata开销通过，才考虑服务与完整WER/SIM。此次仅给出CPU机制和预算，不新增GPU计划。
