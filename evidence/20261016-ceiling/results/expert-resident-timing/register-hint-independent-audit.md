# 寄存器提示与实际资源补充审查

独立确认 resident single/full/zero-debug 三份实际CUBIN的SASS中没有任何SETMAX-like opcode，最高普通寄存器操作数是R29，资源header均为REG32。PTX虽然记录48/160角色提示，但不能据此认定实际每CTA占用53,248个寄存器，或推断只能一个CTA/SM。

之前 `chain-independent-review.md` 的53,248是源/PTX提示的加总，应仅按此理解；静态REG32与实际SASS证据比该提示加总更直接。角色分工和阶段排空的检查仍有意义，但该提示没有证明已执行动态寄存器重分配。没有更改regcap、内核数学或任何已执行的protocol/result。共享内存实际79,488 B/CTA与零spill资源结论不变；occupancy与速度仍需真实测量，不从提示预算推断。

机器证据、CUBIN/SASS哈希见 `register-hint-independent-audit.json`。这是CPU反汇编审查，未新增GPU工作。

父级补充反汇编 `../register-lowering-audit/result.json` 明确查找了实际SASS名 USETMAXREG：vendor native FC1有8条，资源header REG128；TS/resident产物没有对应指令。因此vendor与TS的源级预算不能混比，不能把vendor实际动态重配套用到TS。这里没有新增寄存器cap实验，原定完整调用A/B照常执行。
