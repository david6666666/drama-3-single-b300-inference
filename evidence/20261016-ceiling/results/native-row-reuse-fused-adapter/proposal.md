# Inverse-aware native combine and one slot gather

选择 inverse-aware `combine_experts_add_norm`，并把 compact slot gather 提到本图第一层 MoE 一次完成。相比把三次expand简单合成一次拷贝，这条方案直接去掉展开的读写，且可以用逐AST检查把新计算限制为地址变换。CPU和离线AOT已通过；尚未GPU、计时或改runtime。

原candidate每层5个`index_select`：x、slots、routed/shared/gate展开；40层共200个。新candidate每层仅gather x，第一层再gather一次slots，共41个。40次现有combine+residual+RMSNorm仍各执行一次，其输入改为compact结果，按`inverse[原row]`读取routed/shared/gate；residual和output仍按原1600row地址操作。去掉159个adapter launch，以及每图1,048,832,000字节逻辑expand读写；新增inverse逻辑读取512,000字节。逻辑读写不是实测DRAM流量，不能除以峰值带宽当收益。

上一固定1088全mixed实验距5%门槛，三组还各需减少0.323528、0.314111、0.188109ms。此次有明确可删除的工作，但仍不保证跨过5%，也不声称接近6765fps服务目标。相较旧candidate，只变化adapter地址/slots重复工作；原MoE权重、fulltop8、FP8输入准备、tactics和全部dense精度不变。

`prepare_cpu.py`从冻结e7的原kernel AST生成`inverse_norm.py`，仅增加INVERSE参数与source_row加载，替换routed/shared/gate三处load的row地址。移除这三处地址变换后，函数AST与原函数完全一致：原BF16 sigmoid舍入、显式`fma.rn.bf16`、FP32 residual加和后BF16舍入/存回、WIDTH2048独立行归约、rsqrt、weight1+乘法和BF16输出存储均保留；`num_warps=4`和`enable_fp_fusion=False`也不变。这个证明不替代GPU编译/数值验收。

集成只在新artifact capture范围替换`model.moe_parts`及`mixed_model.combine_experts_add_norm`符号，finally恢复。前者返回compact原生三个parts；后者以inverse间接读取它们。GDN、attention、teacher、model states与所有slots仍独立全量原算。compact slots在第一层收到原row_metadata后生成，各后续层的原row_slots不变，因此不跨图复用旧slot内容。每次replay仍执行这一次gather。同图更换alias/reference时，keep/inverse及prepare元数据一并更新。

为保留新kernel边界证据，40个输出norm缓冲区各有头尾guard，按norm权重对象映射到独立buffer；不使用跨层可变计数器决定语义。compact x每层保活并guard；slot buffer只一份并guard。每个native full mixed control保持原图不变，无候选wrapper。

计划复用上一3case、D64/P1536/span256及compact1088。先完整outputs+111states+CPU/CUDA RNG+新norm/compact/slot/mapguards逐byte比较，包含反转/换reference同图更新；任一失败停并保留。通过后仍仅同一预声明AB100：每个warmup和测量replay恢复同一完整111tensor state，restore在start event之前且双方一致，3warmups每arm，A100→B100，最后统一resolve。计时包含实际gather、inverse-aware combine、backbone和teacher。每case至少降低5%才允许下一阶段，不自动三轮/运行时/服务。

当前CPU结果：`cpu-audit.json`为AST_PASS，helper SHA256 `b391ffa063152915e82415bff03604e1adcd74a0a8956ea7f2a69ac26882d6a1`。完整driver与GPU协议已准备，独立CPU审查无阻塞。AOT实际REG32/STACK0/LOCAL0，两臂staticSHARED1024+dynamic16B；AOT未显式runtime pointer alignment，不声称是实际JIT binary，真实dispatch由完整exact gate验收。
