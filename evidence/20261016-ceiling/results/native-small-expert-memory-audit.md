# 少量专家行：供数、依赖链与连续流水的 CPU 审计

这是源码和已有计数器的机制审计，不是新的服务性能结果。原生计算仍为完整256专家/top8、原始E4M3权重与FP32 scales、K32 MMA/K128缩放边界、原生激活和有序finalizer。NCU node replay 把节点序列化，不能证明生产PDL重叠或把节点耗时直接相加替代完整调用。

## 确定的规模与边界

同一冻结真实 fixture 的 L0/L19/L39 分别有206/159/115个active experts、208/168/146个token8 tiles。512条物理route包含原B64中的63条live及原padding行；每个active expert平均2.485/3.220/4.452条route，token8有效比例30.77%/38.10%/43.84%。低行数降低每份权重对应的有效FLOPs，但不会减少这个expert需要读取的完整投影权重。

每expert的FC1权重2MiB、FC2权重1MiB。三层的最低一份active-expert权重分别为648,019,968 / 500,170,752 / 361,758,720字节（不含小scale与输入）。FC1每token8 tile仍有8个output tiles×16个K128；FC2为16×4。并行原生全局路径已有大量独立工作，旧resident把它们收缩为4CTA、每CTA串行2次FC1+4次FC2完整body，不会凭省中间张量自动加速。

安装源 `batched_gemm/gmem_ab_resources.py:235`按token tile读取expert/mn；`batched_gemm_tasks.py:95`的persistent work循环嵌套K循环，load producer的stage状态跨work延续，末尾才整体tail；因此“增加跨tile预取”并非凭空缺失的功能。`batched_gemm_kernel.py:2985`固定沿M raster；实际HBM物理地址到分区的映射未获证明，不能从expert的大步长就声称partition camping。

## 新 NCU 明确支持和不支持什么

证据为 `expert-m128-ncu-node/L0-native-metrics.csv` 与 `expert-m128-ncu-node-v2/summary.csv`，四进程/八FC节点，每节点4次硬件replay。候选仍是旧metadata-ready版本，**不是后来的route-cache**。

L0原生FC1/FC2 DRAM读432.347/216.412MB，分别接近一份active-expert权重432.013/216.007MB；candidate读432.249/216.297MB。不存在可直接省掉大半重复权重DRAM的证据。L39候选FC1读246.828MB相较原生241.236MB更高，重复tile/不同缓存历史可能相关，但本实验未单独隔离原因。

L0原生DRAM吞吐min/max为76.13–76.23%与68.29–68.75%，没有明显分区吞吐倾斜。若继续考虑分区顺序，应先有分区级证据；现有数据不支持直接grid转置盲测。L2 sector bytes属于不同层级统计，不能用其hit率简单反推出DRAM读量或把1.54倍sector等价字节全算成重复权重。

候选实际active CTA从约1.00增至1.94–1.96，然而FC1 eligible warps L0从.30降至.20，issue22.71%→16.41%，long-scoreboard比值12.54→30.78；L39相同方向（.36→.26、26.83%→20.08%、9.40→24.01）。这说明更多resident warps未直接转化为更多可发射warp。该scoreboard是每issue-active的比值，**不是wall-time百分比，也不是TMA weight饥饿的专用计数器**。TMA由transaction barrier保护；普通LDG结果/地址依赖、scale路径、路由/计数metadata等都需区分。FC2 long也上升，但eligible L39反而.39→.41，不能强行套用FC1同一解释。

vendor每CTA8个weight16KiB stages、约1CTA/SM；candidate每CTA4stages、约2CTA/SM，理论stage容量约同为128KiB/SM、全148SM约18.5MiB。这仅是缓存槽容量，不是实测在途事务：producer是否能及时发射、barrier释放、K-loop长度和TMA/内存系统credit都会影响占用。不能从4<8直接断言供数深度减半，也不能从总容量相同反推延迟完全隐藏。actual vendor shared149,424B/REG128；candidate74,616/74,664B、REG32。PTX寄存器hint不能代替这些真实资源。

## 一个已执行的最小机制检验：FC1 scale route缓存

当前FC1 activation主体也是TMA gather4（route_act1），不是简单的“vendor TMA、Prims普通gather”。安装源 `smem_ab_resources.py:1106`已经每tile缓存TMA gather4行号。另一路 `smem_deepseek_sf_resources.py:295–326`却在每个K128的scale加载中重新读取同一个route_map行号，再通过它计算activation-scale的cp.async地址。原PTX1734/1791/1844等为同地址`ld.global`→地址运算→cp.async，形成可以独立移除的依赖。

新 `expert-m128-route-cache`只在原WAIT之后、每个active work tile prepare内按原valid mask缓存该行号，16→1；无效tile保持skip，不读无效route，不跨tile复用。对512物理routes×8个output tiles，少61,440次有效lane标量读取（逻辑245,760B），字节预算极小；若有收益，应是减少反复依赖/发射，而不是节省大权重DRAM。

已完成CPU/AOT证明：SF K-body七个静态unrolled LDG变零，prepare增加一次，总SASS LDG20→14；原scale/FMA/阶段协议不变；REG32、shared和零spill不变，FC2 CUBIN完全不变。GPU1,516 exact通过；完整call单AB相较vendor三real仅3.042/2.800/4.186%减少，均未过5%继续门槛，保留负结果且不晋级。未另做同轮old-candidate对照，不能把不同运行的差值解释为cache本身的收益。这个结果仍不能分配原aggregate scoreboard的PC来源。

若后续确需归因，最小额外观测应只针对原候选一个真实L0 FC1节点的PC采样，将long-scoreboard样本按（a）SF route→scale地址链，（b）TMA gather准备route，（c）expert/mn/ncta读取，（d）其他指令分桶；保持原完整graph、同输入、原warmups，禁止为了样本重写为独立不同数据kernel。PC采样是否覆盖/足量、counter本身对调度的扰动必须记录；低样本不能当零占比。当前不授权也未执行该新增采样。

## 连续FC1→FC2 pipeline：可实现，但不是旧resident删barrier

旧resident的`ArtifactTask`在每段调用installed `_producer_tail`，`BorrowedSmemAllocator`让阶段复用同一scratch/barrier区。去掉这些drain会让下一段覆盖上一段消费者仍在读的operand、scale或barrier，属于真实生命周期错误。单纯套persistent外循环也不会建立跨FC1/FC2的重叠。

真正的新结构应是一次持久的有界任务DAG：每个expert/token8 tile的8个FC1输出块按原native FP8/scale写入；up/gate成对完成后生产4个128-channel activation块；四块全ready才发布其16个FC2输出任务；finalizer保留原顺序。ready counter、epoch、release/acquire与原输出storage按所有权设计，避免跨replay沿用ready状态。可以保留小global中间区，先不做DSM融合，避免把新调度与新的数值布局混在一起。

关键不是让所有CTA等ready：必须producer-priority调度或保留生产CTA容量，否则FC2消费者占满SM会阻塞尚未完成的FC1，出现持久kernel死锁。不同阶段的A/B descriptor、K长度、TMEM/scale缓冲必须有不相交bank或严格交接，不能借旧alias scratch无条件并发。最简metadata队列上限288 tiles，只有约数KiB counters/ready索引；但两个完整四stage operand banks约145KiB/CTA（再加state/barriers），可能把当前约2CTA/SM降至1，抵消并发收益。按warp/CTA角色分工或共享global intermediates能降低每CTA双bank要求，却扩大调度和发布协议验证面。

因此可验证的收益仅是用已ready FC2工作填FC1供数/尾部空档、避免旧每段反复setup/drain，字节仍约3MiB/active expert。现有node replay不测这种重叠，没有证据支持预报≥10%完整MoE、更不能直接换算服务达标。当前优先级仍应是先定位具体依赖/供数缺口，而不是重写这个大pipeline。

最短完整正确性门禁仍为：CPU证明有界队列/任务所有权/epoch重置与生产者进度；AOT实际资源/无spill；首真实层全FP8/scales/BF16/finalizer逐位；再全部15原生路由场景、partial/empty/replay换map、guard/input/weight/RNG和watchdog。只有完整MoE每个real≥5%、压力场景无>5%回退，才考虑模型与服务精度性能验收。当前这条DAG仅设计，无实现、无GPU结果。
