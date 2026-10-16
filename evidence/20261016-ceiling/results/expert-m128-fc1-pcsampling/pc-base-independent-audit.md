# PC 地址与 pass 字段独立 CPU 复核

原结果 **INCONCLUSIVE 保持不变**。没有重采、没有运行候选、没有修改 frozen extractor、protocol 或原 result。

可获取显式 function base，无需猜测：report 的 `launch__function_pcs` 有一个 instance，值为 **0x100055e6200**；它与 `launch__kernel_name` 的唯一 instance 使用相同 correlation ID 0，名字等于 action 和冻结 reference 的完整 FC1 名。该地址不是最小采样 PC 0x100055e6000。已保存逐字段机器证据 `pc-base-independent-audit.json`。

`IAction.sass_by_pc` 的 installed Python API 位于 `/opt/nvidia/nsight-compute/2026.3.0/extras/python/ncu_report.py:3299`，接收地址。显式base加冻结offset后首段能对应冻结SASS；输出格式存在实际差异：offset0x160 的 BRA 在report中是绝对地址0x100055e6560，在cubin反汇编中是0x360；offset0x180/0x1d0等report省略寄存器`.reuse`标记。因此原先直接把loaded PC当offset以及完整文本逐字符比较都不能得到有效证明。

**这仍不能把本次改为成功。** 使用显式base后，445个有样本PC仅13个符合原严格文本匹配，411个文本不同，21个`action.sass_by_pc`为空。空项合计276个总样本、135个long-scoreboard样本，包含函数前负offset及末端范围外地址；不能静默丢弃或推测归属。没有用文本相似度搜base，也没有为拿到结论放宽旧匹配器。

官方installed `docs/ProfilingGuide/index.html:2794–2804` 区分：`profiler__replayer_passes` 为全部实验中被replay的pass数，`smsp__pcsamp_aggregated_passes` 为统计warp sampling所需pass数。CLI文档 `docs/NsightComputeCli/index.html:1717–1723` 说明max-passes是sampling上限。本report为replayer1、warmup0、aggregated2，命令max1；本机说明未进一步解释这个差异，不能擅自把2除以2或称作两个SMSP域。interval4096、无overflow/drop只是确认的字段，不能消除pass与地址归属问题。

后续若做新的CPU parser版本，应以明确function-PC/identity元数据绑定loaded地址，并单独证明反汇编格式归一化规则；本次全部未知PC和原结果仍需保留。此处不提出额外GPU工作。
