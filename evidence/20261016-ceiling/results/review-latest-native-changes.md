# 两项最新 native 改动的独立审查

审查日期：2026-10-16。范围：`95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b` 与 `e7cda1eba083603b88e1f2e89688e1ff2ced7fc2`；runtime 只读，无 GPU、无测试重跑。源码锚均指上述提交，路径相对于 `drama-3-single-b300-inference`。本报告区分源码正确性、已测可行性与性能推广资格。

## 结论

**未发现这两项 diff 新引入的、能由当前证据确认的并发/槽位/生命周期缺陷。** 两项默认关闭正确。已完成的 exact/state、CPU 边界与完整服务质量证据足够支持保留为实验选项；不足以默认启用或宣称达成所有并发 1.5×。

- queued-admission 扩大了真实帧对覆盖，但一组 C64 服务的 +2.2849% frames/s 同时伴随输出帧数 −2.4241%，不能视为固定工作量或重复性收益。
- private codec Linear cache 的本地优化成立（40 个 cast 消失），完整服务这一组却为 −2.9967% frames/s。应保留负面服务结果，继续默认关闭。
- 6765.407 frames/s 是验收目标。隔离 B64 的约 7734 frames/s 既不是硬件 roofline，也不能证明该目标在完整服务中可达或不可达。

## 1. Queued admission / frame pair

源码：`src/drama3/engine.py:241` 初始化 flags；`:448–490` 单 engine 线程持有 active/waiting/free；`:562–568` 清理取消；`:585–588` 唯一新增选择入口；`:640–693` 新 selector。`src/drama3/frame_pipeline.py:6–14` 保留 capacity、pending/waiting、音频/max_frames 窗口检查。

### 状态与并发

1. selector 只读取 roster/队列状态并更新计数，没有提前弹出 waiting、转移 slot 或触碰 request/model/codec 状态。waiting 与 free 由 engine 线程维护；外部提交只进入线程安全 pending 队列。取消标记可并发变化，但不会让 selector 复用尚未释放的槽。
2. `can_admit=bool(waiting and free)` 后重算 active/batch/deadline 三个条件，与主接纳策略相同，并考虑前一段 cancellation cleanup 后 active 降低。已有可接纳 waiting 会阻止帧对。
3. `Queue.empty()` 是瞬时观察；新请求可在检查后到达。因此这不是“任何新请求都绝不等待第二帧”的保证。对 queued deadline 也同样：刚检查后过期时，下一轮才接纳。代码与服务报告已正确写成**最多额外一个模型步**，不能宣传固定毫秒上界。
4. 音频/max_frames 边界仍要求每请求至少剩两帧。帧对中 first EOS 后第二次模型运算仍会执行，这是既有两步机制，不是本 diff 新增的输出；`runtime_state.py:45–56` sticky done 禁止继续 append，`engine.py:704–711` CPU 停在首 EOS。它会影响全局采样步和下一次调度，不等于单步路径逐请求 RNG 轨迹一致。
5. 完成后先 gather 私有 code/rows 再进入 codec FIFO，旧 slot 状态在提交前不会被下一次 prefill 复用；codec 在同一 stream 顺序读写自身状态。新 selector 未修改这个生命周期。`audio_queue.py:79–105` 的 ready-event、record_stream 与持有 batch 输入仍在。
6. 之前 auxiliary callback 临时 tensor 生命周期问题已有 `frame_pipeline.py:70–75` 的 `self.partial_frame` 持有修复。本 diff 没有修改图或改变这个引用；captured output/hidden2 的固定存储验证仍有效。

### 覆盖与局限

- `tests/test_frame_pipeline_admission.py` 执行生产 selector 的 AST，覆盖默认 off、31/32 running、5/6 batch、deadline 前/等于/后、free 空、pending、audio/max_frames、容量和计数。它是 selector 边界检查，不应描述为完整线程并发测试。
- `pipeline-admission-cpu-and-launch-verification.json`：相关 CPU suite 73 PASS，冻结前文件 hash 与 source 一致；实际 launch 环境只差 deferred-waiting flag。
- `frame-pipeline-production-v2-validation.json` / `frame-pipeline-production-results.md`：图/model 相对 a7118fb 无改动，3689 exact checks、11 cases，包括 full/partial/reversed reused slots、16/24/32/40/48/56/64、max_frames 和合成首帧 EOS；全 model/request/RNG 状态与闲置槽覆盖。合成 EOS 不是自然质量样本，已标明。
- `pipeline-admission-service-results.md`：两臂各1088完成、完整无筛选 WER/SIM、clean drain。4583.673 vs4481.280 frames/s；TTFA p50/p95 反而增加9.762/7.763ms。真实 pair decisions 139/1902→625/1246；计数含64 warmup、排除mixed/prefill分支，不能误称只统计测量窗口。
- 尚无所有并发、三次重复、固定工作量结论；当前不需要为了这个只读审查重复已通过测试。若以后承诺硬时延 SLA，需要专门到达/取消/deadline event 测试，而不是把这个瞬时队列检查解释为硬 SLA。

## 2. Private codec Linear cache

源码 `src/drama3/codec.py:107–116`：只有 `DRAMA3_CODEC_CACHED_LINEAR=1` 时才 deepcopy decoder transformer，并将其中 `nn.Linear` 转到 decoder dtype，随后 compile/capture。缺失、0或其它值均不启用。原有 FP16 decoder 分支不变；本轮合格证据针对 BF16。

### 原生精度与所有权

- `ChannelsLastDACDecoder` 原本把 `quantizer.post_module` 直接交给 `decoder.transformer`。deepcopy 只替换 decoder 引用，原始 codec/reference encode 保留原有对象与 FP32 参数；编译在替换后才发生，不存在先 capture 原参数再换 storage 的悬空引用。
- 当前 checkpoint 有8层×5个无bias Linear，共40个矩阵；FP32→BF16与原生 autocast 线性层的权重转换相同。`nn.Linear.to` 不递归转换旁边 RMSNorm、LayerScale 或残差；17个 norm/16个 gamma 保持原加载精度，编码器与 quantizer共享原模块保持原样。
- decoder 由 graphs 持有，clone 不会在局部函数返回后释放；graph各shape共享不可变 clone参数。服务不提供在线修改这些参数的入口，因此目前无 stale-cache 问题。未来加入热加载必须重建 clone与compiled/captured graphs；不能认为该开关自动追踪原始参数更新。
- clone包括 transformer非参数buffer，不能按仅权重208MiB预算。当前完整服务观察峰值105788→107058MiB（+1270MiB）；是1秒nvidia-smi采样，不是allocator峰值，不能区分所有clone/capture/缓存组成，也不能覆盖亚秒瞬时峰值。现阶段容量足够，但文档不能只写额外208MiB。
- private capture stream 的 workspace隔离未被此改动破坏。缓存参数并不替代这一隔离：异步codec仍须独立capture stream。

### 验证已覆盖与推广边界

- `codec-cached-linear/result.json`：273 exact checks；原始参数名称/dtype/storage/hash前后不变；真实reference-code输入三次推进、全conv/KV状态；B1/B4/B8的F8局部时延减少12.92%/9.23%/8.11%。profile 285→245 kernels，40 parameter cast→0。
- `codec-cache-dynamic/result.json` 冻结生产e7cda1e：46捕获shape PCM/全mutable状态 exact；固定80请求414step trace（不同prefill/decode shape、EOS、槽位复用）normal_sync/cached_sync/cached_poll/cached_thread全通过；观察到的ATen cuBLAS/cuBLASLt ranges无重叠。该workspace检查不声称枚举所有GPU库scratch。
- 动态测试为了差分注入共享恢复状态，生产构造器各server独立状态；固定DP cost table消除分组变量。服务则每次正常初始化重新测DP cost，不能反过来把动态差分的“同trace”承诺用于自然服务。
- `codec-cache-service-results.md`：两臂完整1088均通过WER≤2.9509336013%/SIM≥61.8815893412，candidate相对control WER+0.4187pp、SIM−0.5519，也在用户同源界限内。但4726.771→4585.126fps（−2.9967%），makespan+5.7150%，output frames+2.5471%；这不支持服务性能晋级，也不单独证明可重复回退。
- 该测试使用预编码reference corpus，不是完整`codec.encode`数值回归；原始module参数/对象保持及源码调用关系支持未改变编码路径，仍不应宣称测试覆盖了未执行路径。FP16/其它checkpoint/训练或动态权重修改不在本轮验证范围。

## 3. 6765目标与“理论上限”

冻结C64 vLLM基线4510.271508809941fps，1.5×目标6765.407263214912fps，满64行等效9.460ms/步。隔离真实输入decode8.275183868ms对应约7733.97fps，目标约是其87.47%。这些是**条件算术**，不是roofline。

原因：7734来自一套输入/context/shape/源版本的decode延迟，未同时执行codec、prefill/teacher tower，也未计EOS非音频步、填充、持续接纳低于64的活动槽、首尾排空、传输。kernel与输入变动可令隔离延迟继续下降，所以它不是硬件上限；这些服务成本又会使实际吞吐远低于它，所以它也不是服务保底。不能从6765<7734推出“必然可达”，也不能把两数差额全分配给某个stage再宣告“不可能”。

真正的硬件roofline至少要给实际执行路径逐步FLOPs/读写字节、B300上该dtype/shape可持续算力带宽、并发资源争用和关键依赖；目前证据不足以建立它。对服务目标的判断只能基于同源完整E2E与质量结果，结合阶段时间和active-batch分布估计条件收益。GPU event union不是SM利用率；相互重叠的stage span不能直接相加。当前`assumptions-and-required-metrics.json`和runtime文档最新说明已作此区分。

本轮两项改动与此前GDN正确但变慢的融合，进一步说明局部launch/算子收益不自动转化为服务收益。目标未降低；当前证据既未达标，也没有证明硬件不可达。
