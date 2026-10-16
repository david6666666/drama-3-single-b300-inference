# 原生 MoE 单层流水：CPU 契约审计与最小实验

状态：CPU设计完成；未运行新的GPU任务、未修改runtime、未证明收益。目标仍是所有并发吞吐超过冻结vLLM基线50%，WER/SIM界限不变。先做较低成本的**2×128专家分组、复用原生FI GEMM**可行性，再决定是否投入自写FC1→activation→FC2 tensor-core主体。分组保持全部256专家与每token原生top8，既不裁专家也不增加量化。

工作目录：`/home/zjy/code/hsliu/artifacts/drama3-ceiling-20261016`。CPU复算入口`expert_pipeline_cpu_audit.py`、结果`expert-pipeline-cpu-audit.json`，`cuda_initialized=false`。已有输入和NCU均为历史证据，不能冒充新结构实测。此前`next-native-structure-audit.md`错写FC1为BF16，现按安装版源码更正，旧文件保存在`next-native-structure-audit.before-fc1-contract-correction.md`。

下文FI为安装路径`/home/zjy/code/hsliu/venvs/codex-drama3-standalone/lib/python3.12/site-packages/flashinfer`。模型参照冻结`source-codec-sm-69d636e`；实际native FI路由取`padding-router-b63-native-inputs.pt`，来源冻结6ff8d80，64行中63真实＋1保留padding。只用层0/19/39作为首轮代表，不据此概括全部40层。

## 1. 必须保持的数值契约

模型维度H=2048、I=512、256专家、top8；输入为现有128元素块E4M3/FP32 scale，权重原始E4M3/FP32 scale，MajorK布局，FC1按up512再gate512解释。原生主体是：

1. FC1：FP8输入与原权重block scale GEMM，输出**E4M3字节＋每行每128通道FP32 scale**，不是BF16。`data/csrc/trtllm_fused_moe_runner.cu:440`及`:699`设置DeepSeek、非融合activation、E4M3输出；`:704`调用FC1。
2. activation：用FC1两半的独立scale反量化为float，`silu(gate)*up`；每128通道用CUB max归约，`scale=max(amax/448,FLT_MIN)`，`E4M3(out/scale)`，再保存FP32 scale。源码`data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:236`，scale索引`:311`，计算`:328`，量化`:349`。没有可随意省略的BF16边界，也不能替换成Fish的BF16中间值。
3. FC2：消费上述activation E4M3/scale，用原权重做GEMM，输出BF16，见`trtllm_fused_moe_runner.cu:768`。
4. finalize：按原始k=0..7顺序，用原生路由权重与FC2 BF16值转FP32后累加，只在完整8项之后cast BF16。`trtllm_fused_moe_dev_kernel.cu:680`、向量路径`:842`。随后shared combine、残差、RMSNorm保持现有原生BF16舍入边界。

FC1的具体tensor-core归约/量化epilogue在vendor kernel/cubin里；仅从activation的amax配方，**不能推断FC1量化的舍入过程已经等价**。自写主体第一门必须对真实FC1 bytes和scale逐项比对，不以最终shape/dtype相同代替证据。即便公式相同，FMA顺序、expf实现、FP8饱和转换可能产生差异；保留差异报告，不能自动标注无损。

## 2. 无需修改参考C++即可暴露FC1/activation

安装版已有`DeepSeekFP8DABodyBuffers`稳定7项ABI（`data/csrc/trtllm_fused_moe_kernel_launcher.cu:369`）：

| 索引 | tensor | native类型 |
|---|---|---|
| 0 | gemm1_output | uint8保存E4M3字节 |
| 1 | gemm1_output_scale | FP32 |
| 2 | activation_output | uint8保存E4M3字节 |
| 3 | activation_output_scale | FP32 |
| 4 | gemm2_output | BF16 |
| 5、6 | workspace_fc1、workspace_fc2 | native scratch |

`fused_moe/core.py:1721`的`runtime.DABodyRunner.prepare_body()`分配这7项，`forward_from_metadata()`使用固定tile/tactic和调用方保有的buffer运行完整native主体。通过`runtime.MoERunner`构造DeepSeekFp8、top8、H2048/I512、256本地专家，原生canonical routing＋对应tile metadata，拿到参考中间值。调用只在GPU授权之后进行；本次仅阅读源码。

限制和索引：

- C++ `:4296`、`:4325`显式要求`do_finalize=True`。它可用来导出完整参考以及分区各自中间结果，**不能直接当无finalize候选**。分区诊断可单独用do_finalize=True导出中间值，但其partial finalized BF16绝不作为最终组合输入，且不计入候选计时。
- scale buffer分配形状是`[8,max_capacity]`、`[4,max_capacity]`，实际kernel按`permuted + actual_total_padded_tokens*channel_block`寻址。诊断要从metadata读actual padded数，对flatten的实际有效前缀重建索引；不能用Tensor逻辑stride直接索引。图内不能`.item()`制造同步。
- 只比真正有效expanded→permuted行的中间输出；未参与padding字节可能未初始化，不能把它们的不同误判为模型误差。另行检查状态/边界canary。
- reference普通logits路径与prepared路径必须先在相同tactic下相互比对，证明导出接口没有改reference。使用原生router的int16 IDs与BF16 weights；不使用torch.topk/softmax重算作为精确参考。

## 3. 优先最小候选：两组专家复用原生FI

假设：固定同一B64层的global routing，分组0的FC2可与分组1的FC1重叠，降低全体专家之间的阶段等待；在重复routing metadata/activation launch和更大静态grid成本后，整个单层仍更快。**不是减少weight bytes，也不是承诺两个饱和GEMM叠加后免费。** 原native已启用PDL时也有部分阶段重叠，新结构必须与同PDL基线比较。

可调用API：`fused_moe/core.py:5918`的`trtllm_fp8_block_scale_routed_moe`。支持global num_experts=256/top_k=8，local范围0..127、128..255；`data/csrc/trtllm_fused_moe_kernel_launcher.cu:1386`验证范围，RoutingKernel.cuh:590按global ID过滤，GEMM权重索引为ID−offset。当前模型shared expert单独执行，native不支持fused shared expert+EP的限制不妨碍本实验。

最小数据流：

1. 原有prepare_moe_input的FP8 bytes/scale及router logits保持不变。在主stream只做一次native canonical routing，完整256专家、routing_method_type=4、top8、scaling1、norm_topk_prob原值、use_routing_scales_on_input=False、原PDL。保有IDs/weights稳定地址。canonical IDs为int16，使用无损int32转换送公开unpacked API，并将这次转换成本计入live-routing计时。不要假定canonical packed scratch可作为稳定公开格式。
2. 两个独立stream等待routing-ready event。各自调用原生routed API，传同一global `(ids_int32, BF16_weights)`，local_num_experts=128、offset0/128，w13/w13_scale/w2/w2_scale沿expert轴切连续view，保持原字节与scale。没有per-partition renormalize，也不把num_experts改为128。
3. 每组`do_finalize=False`返回本组FC2 BF16和expanded→permuted map，非本组expanded项为−1。**每个全局512条route必须在两图中恰好出现一次**；有效map必须在本组有效padded区间，不能假定两组permuted顺序相同。公开Python封装会把借用的native weights补回返回值；直接FFI的该slot可能为空，见`shared/inputs.py:171`，所以finalizer显式持有canonical weights，避免盲做dlpack。
4. 主stream等待两组done，只运行一次定序finalizer：对每token/hidden，k=0..7依次按两张map选正确的FC2 BF16，再按原FP32乘加顺序累加、最后cast BF16。不能先分别形成partial BF16再相加，也不能使用atomic_add或树形归约。最小artifact可以写一个清楚的CUDA/Triton consumer；是否fused FMA必须与native实际结果对照，不能仅靠源码表达式猜测。先对单组256的`do_finalize=False`重建精确finalize，再用于两组。
5. 本轮不改变shared expert排程、不融合后续RMSNorm、不增加第三候选配置。这样能清楚判断专家分组本身的净贡献。

workspace与graph：两个stream各自持有完全独立routing scratch、FC1/activation/FC2 scratch和输出，不能共用DA max-workspace（该API原本只承诺互斥body共享）。参考图、serial partition图、parallel partition图用独立pool；同一个parallel图内部以event依赖描述fork/join，不靠host同步。权重只读共享。输出和scratch存活至主stream finalizer结束；每次replay不得分配新地址或偷用上次map。每组先warmup/固定native tactic再capture。local128改变valid tactic集合，须检查该配置有效，不能直接照搬不合法的global256索引；若只有不同tactic可用，记录为结构引起的必要适配并逐阶段验证。

### 成本与并行度：为什么先2组，不先4组

原NCU三个B64层的实际FC1 grid为(8,288,1)=2304 CTA，FC2为(16,288,1)=4608；512threads、128registers/thread、149424B shared，register/shared都限制1 CTA/SM。设备148SM。不能把当前grid约说成1300 CTA：1300更接近部分层的非空FC1估计。

固定tileTokens=8时，源代码`data/include/flashinfer/trtllm/fused_moe/runner.h:101`给出最大batch grid：`min(E,512)+floor(max(512−E,0)/8)`。GEMM使用local专家数（`trtllm_fused_moe_runner.cu:238`），但routing workspace容量仍按global256（launcher`:1428`）。由此：

| 分组 | 每组batch grid bound | FC1总CTA | FC2总CTA | 静态grid增长 |
|---|---:|---:|---:|---:|
| 1×256 | 288 | 2304 | 4608 | 0 |
| 2×128 | 176 | 2816 | 5632 | +22.22% |
| 4×64 | 120 | 3840 | 7680 | +66.67% |

这是源码推导而非新GPU实测，包含提前退出CTA。activation仍以64×8全部expanded entries产生grid；当前148SM下heuristic选2token/CTA，(4,32,8)=1024 CTA，2组变2048、4组变4096，不同组的无关项早退。额外routing/padding及launch成本可能抵消重叠。

真实FI IDs，63real＋1padding：

| 层 | 活跃专家 | token8组 | 两组route数 | 两组token8组 |
|---|---:|---:|---|---|
| 0 | 206 | 208 | 250 / 262 | 106 / 102 |
| 19 | 159 | 168 | 227 / 285 | 82 / 86 |
| 39 | 115 | 146 | 206 / 306 | 61 / 85 |

层39若4组则route111/95/219/87、token8组35/26/52/33，明显不均衡。两组是最小可行测试；动态任意expert重排或按负载分组会引入额外gather/weight重排和语义验证，本轮不做。真实64行旧输入的route census来自torch.topk复算，tie等价未证，另存在JSON中，只作并行度估计。

## 4. 后续自写cluster主体的最小资源设计

仅在原生分组不足、而中间值导出已经明确后考虑。每expert/token8为一个4-CTA cluster，每CTA负责up128＋gate128通道，形成其activation128片段；cluster同步后每CTA可读完整activation512，每CTA计算FC2的512输出通道，分128输出tile流式执行。这样既不重复4次FC1，也不把FC2跨CTA部分和改变为新归约。

逻辑工作集（估算、未编译）：

| 项目 | 每CTA bytes |
|---|---:|
| FC1 Ktile128、up+gate128、双buffer权重 | 65536 |
| 输入token8×K128双buffer | 2048 |
| FC1 FP32 accumulator token8×256 | 8192（寄存器/TMEM，不计下方SMEM） |
| FC1 E4M3 bytes＋FP32 scale | 2048＋64 |
| 全activation512副本＋scale | 4096＋128 |
| FC2 outputtile128双buffer权重 | 32768，可复用FC1存储 |
| FC2 FP32 accumulator token8×128 | 4096 |
| 初步SMEM峰值，未含swizzle/barrier/TMA descriptor | 73920 |

256threads时单看FC1逻辑accumulator为每thread8个FP32寄存器，这不是kernel总寄存器需求。Blackwell MMA最小M/N形状、TMEM分配、producer warp、TMA alignment及cluster占用会扩大资源，编译后才能确认。设备每block shared上限232448B、SM233472B、65536registers；4CTA cluster跨SM，不能拿单SM共享上限当cluster可以任意共享的证据。activation分片用DSM或显式复制并配合cluster同步；普通CTA之间不能直接读彼此shared。

单expert FC1+FC2原权重合计3,145,728bytes（3MiB），每CTA约786432bytes，不可能整专家权重常驻73KiB。保留的是activation生命周期，权重仍走TMA分块。若某expert多于8路，会有多个token8 cluster，权重可能重复；要记录实际DRAM读量，不能假定“resident”即只读一次。

一CTA/expert并行度不足：实际FI层39仅115活跃专家，最多115/148=77.70% SM，即使每token8一CTA也仅146个约一波、负载尾部长。4CTA cluster对应层0/19/39为832/672/584 CTA，按148SM约5.62/4.54/3.95波，实际cluster放置限制还可能降低并发。与native数千CTA相比仍少，必须证明确实能维持带宽，不能以融合数量判定优劣。

逻辑中间流量预算也限制收益：按层0真实FI的1664 padded行计算，FC1＋activation字节/scale的全局写读合计约5.27MB，活跃专家权重约648.02MB（不含weight scales），前者约0.81%。activation实际仅写真实routes且缓存会改变DRAM量，此比值只是逻辑容量估计。**仅删除中间DRAM往返解释不了大幅GEMM提速**；需要证明新的跨阶段供数/尾波缩短提高了真实带宽或减少调度空洞。旧NCU六GEMM读量已约一份活跃专家权重，DRAM56–75%peak，因此不能宣称消除大量既有重复读取。

## 5. 一次GPU可行性的预声明与门

GPU授权后先只运行层0固定实际B64输入，必要的层19/39作为负载分布差异验证；不直接改生产模型或跑服务。冻结artifact/hash、固定GPU UUID/原source/model权重、dtype/PDL/layout/tactic、输入routes，所有输出和失败独立保存。

三个必要控制共用同一输入与原路由：A当前完整native MoE；B两组serial执行＋同一个定序finalizer；C两组parallel执行＋同一个finalizer。A→B包含分组与额外调度成本，B→C才隔离并发收益；C必须超过A才能继续。另报含input quant/native canonical route/IDs转换的端到端层耗时，以及纯body阶段，不能只展示较容易变快的一项。tactic warmup/capture不进入计时，先做固定轮数交错replay并给分布，不扫一堆策略挑最低点。

正确性门按顺序：

1. 普通reference↔prepared reference逐项FC1 E4M3/scale、activation E4M3/scale、有效FC2 BF16、routed BF16精确；native routing IDs/BF16 weights及输入FP8不变。
2. 单组do_finalize=False→新finalizer精确，换routes多次replay也精确。再验证两组每route恰一owner、每组非local map为−1、有效map无越界，serial/parallel每路FC2及完整routed输出精确。tactic合法但出现数值差异时量化maxabs/RRMS、分清阶段，保留FAIL，停止性能晋级而非默改阈值。
3. 所有输出完整，输入/权重/canonical weights不被写入；不同iteration路由改变、空local专家、单组极偏、padding和canary检查；并发读写无workspace别名。纯MoE无模型状态/RNG，进入full model后再覆盖真实/非参与state和RNG。
4. 以未profile墙钟event对C相对A的完整层净收益决定是否延伸，root授权首门≥5%且层19/39不出现明显回退。该5%是投入下一门的工程阈值，**不是用户目标降为5%**。若图中FC1/FC2根本没有重叠，保留trace解释stream/资源限制，不用计时噪声宣称流水成功。
5. 有净收益才进入40层单/多步native状态与EOS/cancel/slot生命周期，再同冻结源完整1088服务+WER/SIM；正式吞吐结论仍需重复及全部并发。单层份额旧pure GEMM3.957ms/activation.356ms和mixed7.700/1.475ms只是覆盖工作，并不是可以直接相加的服务收益。

停止本轮坏候选不等于停止用户目标；若分组额外成本吞掉收益，用实测中间值与trace继续决定cluster主体、shared-expert依赖或GDN/prefix结构下一步。当前目标继续INCONCLUSIVE。
