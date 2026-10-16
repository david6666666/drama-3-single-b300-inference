# C128 原生前缀 DAG：最小片段接口与停止条件

这是 CPU 设计，未实现 kernel、未启动 GPU、未修改 runtime。原 exact FAIL / local continuation FAIL 均不变。本方案只复用**当前已到达且已被同一 admission 选中**的相同前缀，不等待更多请求，不预热未来输入；跨轮 completed-cache 暂不纳入。引用源固定在 `source-codec-cache` 的 e7cda1e；既有 CPU 预算及语料证明见 `../prefix-dag-cpu-audit`、`../reference-prefix`。

## 先做一个有边界的 GDN 片段

首个候选只从真实层的 **post-convolution q/k/v、decay、beta** 开始，返回原 GDN 输出与最终 BF16 state。保持 dense projection、conv、attention、MoE、teacher、sampling 全部在实验外。首组可取已记录的请求1040/1041：长度191/178、共享前缀156，切在128；之后才扩到真实30层和其它owner/branch分布。它先验证整个 DAG 最容易错误的 FP32 recurrent carry，不冒充完整模型优化。

建议 artifact 接口：

```python
plan = C128Plan(
    original_lengths, original_slots, owner_of_request,
    prefix_cu, suffix_cu, prefix_chunk_indices, suffix_chunk_indices,
    prefix_chunk_offsets, suffix_chunk_offsets,
    original_to_compact_row,  # CPU验证所有实际行；不按输出值重排
)
workspace = C128Workspace(
    private_h32,             # [request_capacity,32,128,128] FP32
    prefix_h_bf16, suffix_h_bf16, prefix_updated_v, suffix_updated_v,
    prefix_output, suffix_output, final_state_bf16,
)
run_gdn_c128(native_inputs, plan, workspace)
```

`owner_of_request[i]`只在真实frame bytes `[0:129)`相同且已知公共prefix P>128时允许指向别的请求；复用已有完整prefix hash并做字节确认更保守。P≤128、版本不同、非零初始state不一致、输入缺失均不合并。decode行和padding行不得加入request别名。首次片段用已有完整原生行输入建立索引，不改变投影batch以免同时引入不同GEMM舍入。

**阶段顺序改为 unique-prefix → 所有 suffix，而不是 owner-full → branch-suffix。** 原 H 的 BT=64（`gdn.py:130–180`），191/178各有3个串行chunk；owner先完整3chunk再branch tail1chunk会得到4chunk依赖深度。先唯一128行2chunk、再两条tail各1chunk可保持3chunk深度。这里仅是依赖图，不是延迟预测；额外launch、fork和小grid仍可能更慢。

第一阶段运行每个唯一prefix的原生 l2norm/cumsum/intra/H/O；第二阶段运行所有owner自己的tail和branch tail。后续完整DAG可以把各片段的独立l2norm/intra合并调度，但不要在第一份证明里另写融合核。为避免长的未复用请求阻塞tail阶段，最终实现应把所有>128的已到达请求都按128切段（不复用者有独立prefix），≤128请求仅进入第一阶段。这个切分本身也是待测成本，未重复者不能因分段被计成节省行。

## 保留 FP32 carry，不能恢复最终 BF16 state

installed `sglang/.../fla/chunk_delta_h.py:130–131` 把initial_state同时用作ht；149–165加载后转FP32，167–328在chunk间保留FP32，171–186仅将供O读取的h舍入BF16，331–348按最终state指针dtype存储。当前runtime state是BF16。若prefix阶段先写BF16再继续，会新增原模型没有的中间舍入。

最小实现直接调用原H核，以**私有FP32**数组作prefix和tail的in/out，原BV32/4warps/2stages、dot顺序、exp实现、BF16 h/updated-V边界不变。prefix结束后，把owner FP32状态复制到其branch各自私有槽；owner自己的tail可以原地复用owner scratch。所有fork完成后才运行tail；tail结束后单独按原最终边界转换到请求BF16 state。scratch不与真实state/KV/其他decode slot别名；每层和每次replay初始化prefix零状态，不能沿用上次tail。

原 `TRACK_STATE` 可在完整控制请求 i_track=2 时导出参考FP32 checkpoint（188–207），但对恰好128长度、NT=2的prefix核，循环永远不进入i_t=2，**不能借这个hook导出结束state**。候选应依靠FP32 ht。FP32指针会产生另一编译实例；0+checkpoint可能涉及signed-zero/FTZ；拆开kernel也可能影响编译。因此这些都必须实际exact gate，源级“同公式”不等于已精确。

## 第一片段的门槛和最小改动

只新增 artifact wrapper/plan-checker/driver，不动installed FLA或production。直接复用原l2norm、cumsum、intra、H、O；显式设置原source环境，不做tile sweep。控制full H加TRACK的观察分支也必须先证明所有原结果/state不变。比较：

1. prefix的FP32 checkpoint与控制TRACK逐bit；原chunk BF16 h、updated-V、最终GDN output逐bit；所有实际请求最终BF16 state及未参与slot逐bit。
2. 同一captured graph更换长度、owner映射和slots；含不复用、P128 fallback、P129、尾长1/63/64/65、空尾、privatepadding、owner自己也是tail。原始数据+期望metadata只在graph前更新，保活到GPU完成。
3. 第一处差异保存tensor和元数据后停止晋级；不把数值容差自动代替状态/索引bug。若有独立native算术重排理由，需另声明诊断协议，旧FAIL保留。
4. exact通过之后，单独计入gather/zero/fork/publish/全部kernel；再考虑完整层。不存在GDN片段局部通过即可服务推广的捷径。

## 扩到完整prefill所需的明确接口

- **逐层一次权重扫描**：紧凑隐藏行表示为decode + unique prefix + 每请求tail。每层dense输入投影、out投影和完整top8 MoE仍各一次；不把prefix全部40层跑完再跑suffix40层。残差/归一化跟紧凑实际行，prefix输出归其owner；最后只展开用户需要的state/KV和末行hidden。
- **conv**：`packed_prefill.py:109–146`目前负source补0。新suffix的前三个history访问应解析到owner该层projection的原125:128行，仍按原四tap顺序和BF16→SiLU边界；写final conv history到每请求独立slot。projection所有行已就绪，conv不必等待prefix GDN。stride必须沿用实际packed projection，不能假设8192连续行。
- **attention**：`mixed_model.py:81–92`当前q/k/v使用同一boundaries。需要显式 `q_cu/kv_cu`：unique-prefix query看prefix KV；每请求tail query看owner prefix + 自身tail KV，绝对RoPE位置保持128+t、causal为key≤128+t。KV已由本层一次projection得到，无需等待prefix attention output。可一个packed native call表示这些逻辑sequence，但ATen unequal-Q/KV causal对齐及算法选择必须实证；恢复KV后照旧调用当前接口会漏prefix。不实现通用paged/radix cache。
- **teacher**：每行label仍取原请求下一行，C<P保证共享第127行label128仍公共。13层tower KV同样需要prefix+tail语义；保持first-frame与旧decode tower现有采样顺序，sampling_step不因共享改变。
- **调度/寿命**：沿用原admission，旧decode只推进一次。每次graph所有prefix/branch metadata和私有scratch由同一owner持有，slot复用等completion；不发布跨轮ready cache。`MixedGraph.forward`现有prefill reset和原行metadata不能直接照搬到紧凑结构。

这意味着完整实现至少触及独立DAG layout、GDN wrapper、conv地址解析、attention Q/KV layout、mixed backbone及teacher wrapper；它不是一个 `gdn.prefill(initial_state=...)` 的单行优化。先证明GDN片段才能决定是否值得这组改动。

## 预算更具体，但没有新的性能结论

`result.json`从已冻结FIFO admission代理重算198组：原真实请求3452个64行chunk，C128 DAG为2872，减少580（16.80%）；可共享37120prompt行（20.064%）。不同数字源于最终partialchunk仍须计算。它们不含padding/olddecode，且仍不是实际记录的request-ID命中轨迹。

代表组190：8请求/4owner，原24chunk→16chunk；原1354prompt行→842行。组192：8请求/5owner，原24chunk→18chunk；1335prompt行→951行。两者不是当前MoE任意prefix+common25复用的661/531别名；C128只能删512/384行，不能混用更大收益。

scratch只需每个请求单层FP32 carry2MiB：8请求16MiB，owner state在fork后可供自身tail原地更新；跨30层复用同一scratch。每个branch的fork逻辑read+write至少4MiB/层（30层120MiB），另有prefix零填、FP32 final publish至BF16、h/O中间张量与元数据。290个branch累计fork逻辑读写约33.98GiB/1088请求，**不是DRAM实测量**，不按标称带宽换算秒。完整跨轮快照67.15625MiB/entry不是本同层scratch需求。

历史 `mixed-profile` dfdd598 的单58decode+6arrival、D64P1280span256图三次profiler记录，H合计每图0.966746ms、O0.978267ms，含l2norm/cumsum/intra的GDN前端合计3.371632ms；conv1.259080ms。该trace早于compact-GDN等改动，不是当前收益权重，也不是相同复用case。只说明首片段不能把完整mixed约28ms都计为GDN收益；已有H/O fusion全图负结果也不能被前缀设计覆盖。

同组C128沿用先前偏乐观线性预算约0.992447s/22.895377s窗口（4.335%）；把olddecode及固定权重扫描也按行比例减掉了，尚未扣上述成本，既非实测也非严格上界。已改善MoE复用属于其中，不能再次相加。仅自然完成cache的71hit不是同组290branch，本方案未把它们算进来。当前6765目标所需同ledger减少7.35345s，C128没有可证明单独填平缺口的预算。

**结论：先做一个真实GDN两请求C128原核分段的精确性片段是可实施的；完整prefill DAG仍有attention/conv/teacher接口工作和有限时间预算。此CPU提案不支持立即生产接入或宣称目标可达。**
