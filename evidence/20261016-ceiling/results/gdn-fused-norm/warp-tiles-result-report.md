# C3 四个原生 V32 warp 融合：精确通过，性能失败

**结果：停止晋级，不修改 runtime。** C3 的 explicit-layout 方案恢复了真实输入的原生逐位结果，但组合算子耗时比原实现慢6.37%。后续一次CPU生命周期变换也未达到寄存器门槛，未运行GPU。

| 检查 | 结果 |
| --- | --- |
| 此前C2失败的真实首层fixture | 原生A复现精确；C3 state/raw/norm/unused全部bitwise exact且finite |
| 30层×6批次条件 | 全180用例精确；B64/64、B64/58、B48/40用C3；B32/8/1维持原BV16路径 |
| Debug raw输出及production no-dump | B>32两种specialization均通过逐层精确检查 |
| 完整图 | 六组条件各8步，codes及全部111个状态张量均精确，包括EOS/标量状态和未参与slots |
| 局部计时 | 原生1.018777580ms → C3 1.083673439ms，慢6.36997%，未达到≥5%改善门槛 |
| 计时中的状态/输出 | 每轮结束均继续bitwise exact |

计时冻结source463d837，GPU7；取真实prefill+8decode后的30层输入，slot重编号为紧凑64池。计时前对这个重编号版本再做状态/输出精确检查。两组都用CUDA graph计完整 recurrence+z.contiguous+gated_norm；C3删除对应复制/独立norm，使用production no-dump。每组6个样本，AB/BA交替，每样本100replay、10warmup，initialstate重置在event之外。输入固定，重复replay后的state仍逐轮精确；这是组件测试，不是服务吞吐或全请求生成测试。明细见 [warp-tiles-timing-result.json](warp-tiles-timing-result.json)。

GPU进程审计见 [warp-tiles-process-audit-summary.json](warp-tiles-process-audit-summary.json)及原jsonl。三个执行阶段target GPU同时最多一个进程。原C/C2的EXACT_FAIL、NUMERIC_FAIL保留；C3没有改变其数值门槛。Expanded driver先后有CPU语法和标量Long→Byte view的仪表错误，已单独保存initial-syntax及scalar-error记录；修复仅影响比较器/打印，候选源码hash未变。

## 编译资源解释与限度

Debug C3是177registers/thread；**实际计时的production no-dump是182registers/thread**。这个差别已经通过实际runtime cache的cubin resource dump确认，不只依赖AOT。两者均LOCAL=0/STACK=0，Triton requested shared512B，cuobjdump分配报告1024B。完整cache路径/hash在 [warp-tiles-resource-audit.json](warp-tiles-resource-audit.json)。

SM的64K寄存器下，production182×128×3=69,888>65,536，所以原始寄存器容量本身就把C3限制为最多2CTA/8warp；原A168×32的单warpCTA容量上限为12warp。分配粒度和其它约束只会降低这些上限。这里没有NCU occupancy测量，不能把它写成实际occupancy或宣称已唯一定位回退原因。额外conversion/同步、CTA调度粒度也可能影响性能。C3仍保留原state主读写流量；少两个后处理kernel并不保证抵消驻留warp减少。

## 只做一次的CPU生命周期审计

C3 PTX把Q/K从每lane4个值转换为矩阵slice的每lane8个值，发生在581–624附近，早于decay和state操作。原生A仅在第一次h·K之前（879附近）转换K，且在rank-one更新之后（1576附近）转换Q。C3的Z/WEIGHT基址还从66/67活跃到2390/2410，各为64bit；它们确实延长了epilogue元数据生命周期，但这只解释有限的地址寄存器，不能直接归因全部14个额外register。

因此只尝试一个有源证据的CPU变换：将Q/K layout conversion延后至原生首次使用位置，源级算术、tile、dtype不改；目标是保留已审计的K/FMA树。新文件 [fused_norm_warp_tiles_late.py](fused_norm_warp_tiles_late.py) 与 [warp-tiles-late-cpu-compile.json](warp-tiles-late-cpu-compile.json) 独立保存。编译后的shuffle确实延后，但production寄存器数 **182→185**，依然无spill。源级生命周期缩短并未降低最终分配峰值，所以资源前置门槛失败，未继续完整PTX等价审核，也未进入GPU精确/计时测试。

把Z/WEIGHT/OUT基址放到只读metadata表并延后加载，仍需让表基址跨recurrence存活；净收益估计最多2–4个32bit地址register，还会多2–3个64bit加载。stride在当前specialization里已是编译常量，不能再靠移除stride参数获得寄存器。没有证据认为仅这项能把185恢复到≤168；因此没有继续实现metadata间接寻址，没有maxnreg cap、spill换占用、tile sweep或浮点重关联。

此方向现在保留为“原生精确但收益失败”的经验。现阶段应优先完成有证据的codec参数cast缓存等其它路径。完整服务1088/WER/SIM未针对C3运行，因为本地性能门槛已经失败。
