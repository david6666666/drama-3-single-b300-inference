# GDN fused prefill 两个变体的独立 CPU 审查

**未发现推翻两者 local continuation FAIL 的实质精度或计时错误。** 已保存结果、原始精确检查数、冻结哈希和三次计时复算一致；两者继续保持不晋级。此审查只读取源码、已执行artifact与CPU反汇编，没有重跑GPU。

机器复核见 `independent-review-checks.json`。审查对象是 `gdn-fused-prefill`（O BV32）及 `gdn-fused-prefill-o64`（O BV64），原生source e7cda1eba083603b88e1f2e89688e1ff2ced7fc2；本轮证据发布commit f1dc31d。原始已执行protocol/result保持不变。

## 原生精度边界与覆盖

`fused_ho.py:41` 的H先转换到BF16，再供O计算；`:55` 的updated值转换BF16且invalid token填零；`:49`保留两个K64 prediction dot顺序，`:64`保留BK128的O计算，`:72`保留原生缩放和BF16中间量。H recurrence使用原始updated并按原native边界再舍入。它没有拿更高精度的局部值绕过原生BF16 H/updated接口。

synthetic覆盖长度128/107/0/1/63/64/65、256/193/17/0、333/1/129/128，非零initial state、slot=-1、额外untouched state；debug H/updated、production输出/state和capture/replay均逐字节比较。空序列和O尾部padding没有被从检查中删掉。

`validate_real.py:54`保存全部111个状态tensor与RNG；`:57`逐层probe在相同输入/state克隆上对比，并返回native结果给后续层；`:67`要求30个真实GDN层都访问。三个真实case D58/P6、D40/P3、D12/P1共90层case。`:68`及`:70`另外恢复全状态/RNG后对比完整mixed graph，覆盖候选误差在网络中传播后的输出及全状态，不仅是单层教师驱动检查。

每个变体已记录57项synthetic、522项real、1026项timed-trajectory精确检查，机器复核所有布尔结果通过，未使用容差放宽。完整图比较包括30个conv、30个GDN、23对KV及request/sampling状态。没有全1088服务质量验收或性能晋级声明。

## 计时公平性与复算

`time_real.py:54–68`在双方capture/prepare和预热后，对每个计时arm恢复同一base state/RNG并同步，恢复成本在event外。三次顺序AB/BA/AB，每次连续20个graph replay；每个20步轨迹结束后再次比较输出、所有state和RNG。双方起点公平，但这是固定20步轨迹，不是每个replay独立reset，也不是真实请求动态调度。

| Case | O32 native ms | O32 candidate ms | O32变化 | O64 native ms | O64 candidate ms | O64变化 |
|---|---:|---:|---:|---:|---:|---:|
| D58/P6 | 26.355579 | 26.490366 | +0.511417% | 26.355219 | 26.420694 | +0.248435% |
| D40/P3 | 18.262636 | 18.286597 | +0.131202% | 18.257458 | 18.255545 | −0.010480% |
| D12/P1 | 15.425816 | 15.449295 | +0.152204% | 15.431131 | 15.429988 | −0.007406% |

三次中位数及比例与原summary在1e−15误差范围一致。两者都未达到已声明的所有shape continuation gate；O64另两项接近零的变化不能写作可用收益。计时未包含codec、HTTP、scheduler和完整采样服务，不支持外推6765 frames/s目标。

## 资源与结论边界

O32 production为245 registers、115,492 B dynamic+1,024 B static、TMEM256；O64为206 registers、132,884 B dynamic+1,024 B static、TMEM512。CPU/runtime所有受检CUBIN的STACK/LOCAL为0，已记录runtime n_regs/n_spills。CPU与runtime的specialization可产生不同CUBIN，原文没有谎称相同。O64减寄存器同时增shared/TMEM；V32重复QK的结构可从源码确认，但没有把这写成已量化的SM stall归因。

一处应收紧的证据措辞：进程记录只证明GPU0在pre/post采样时没有外部进程，不能证明整个窗口持续排他。已通知root，root将修正report/analyze里的exclusive措辞；不补造不存在的持续审计，也不改原始protocol/results。这项限制不改变当前负结果、不晋级的结论。
