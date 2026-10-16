# Resident chain 验证器与资源审计：CPU 阶段

本报告仅说明 harness 已准备和 CPU 合同通过，**主链 kernel 尚未进入本验证器，未运行 GPU，也未证明完整驻留链正确或提速**。GPU gate 由 `/root/vllm_scaling` 统一冻结、调度。

## 已完成

- `validation_contract.py`：15 个原生 oracle 的 shape/dtype、原始 metadata 双向映射、expert tile、绝对 mn_limit 检查；原始 top8 顺序的 CPU FP32 FMA 诊断结果逐字节匹配全部 archived native `routed`。这不是新的 GPU finalizer。
- layer0/expert205 的有效行 0–8 fixture：inactive q/input scale 归零；所有输出的无效区预置独立 sentinel。FC1/activation scales 按 block/token 定义，输出必须直接符合原生逻辑顺序，不允许 checker 逆排列。
- `validate_chain.py` 提供 `run_single(build_chain, directory, manifest, gpu_uuid)`：eager exact 后才捕获 graph；检查五个阶段边界、所有只读输入、weights、RNG、填充输出；原始失败存盘并立即终止。`candidate.owned_tensors` 仅保活，明确不将 mutable scratch 当只读；`candidate.immutable_tensors` 单独声明。
- `run_full(build_full_chain, load_weights, ...)` 预留完整256专家路径：每层同一地址上的 graph 连续切换五种原始 metadata；CAP2304 / CTA288；输出scale采用runtime padded stride。五个中间边界和最终 routed 全部 exact，padding/tail保留sentinel。
- finalizer 使用已安装的 `trtllm_moe_run_finalize`，ABI来源 `flashinfer/prims_ts/moe/runner.py:2383–2397`，保持原始BF16 route weights、expanded top8次序和最终BF16边界。单expert阶段用候选分支替换原生其余分支后调用该函数，明确属于 **oracle-assisted composition**；不能称作完整MoE实现。
- `resource_audit.py` 从 host IR 中 `CUlaunchConfig.sharedMemBytes` 的实际 store 读取dynamic shared，再加CUBIN static，检查≤80KiB、LOCAL/STACK0、PTX/SASS无spill。拒绝未知或歧义host IR，不用allocator估计替代实际资源。cluster4必须另给实际launch证据。
- 资源解析器仅用既有FC2编译产物做CPU自检：正确读出72,616动态 + 2,048静态 =74,664字节。该结果只证明解析器能处理当前编译IR，不给新resident kernel资格。

CPU命令：

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 /home/zjy/code/hsliu/venvs/codex-drama3-standalone/bin/python validate_chain.py
```

`cpu-contract/result.json`：CPU_CONTRACT_PASS，15 oracles / 9 partial fixtures，CUDA未初始化。`resource-parser-component-check/result.json`：已知component解析自检PASS，cluster1；不能冒充cluster4链路资源结果。

## 协作 ABI

单expert build输入：`q[8,2048]E4M3, input_sf[16,8]FP32, w1[1024,2048]E4M3, sf1[8,16]FP32, w2[2048,512]E4M3, sf2[16,4]FP32, valid_rows int`。

返回对象要求 `launch(CUstream)`、`outputs`、`immutable_tensors`、`owned_tensors`。`outputs` keys固定为 `fc1_bytes,fc1_sf,act_bytes,act_sf,fc2`。debug输出必须互不重叠且不得覆盖只读输入。全部owned对象保持引用直到graph结束。

完整routing build另接收 q/input_sf、全部256权重和原始ids/route_weights/tile/mn/route/ncta/padded/mapping/valid。harness逐case原地copy payload，地址不变；kernel不得缓存先前Python metadata值。full输出rows为CAP，scale为flat8CAP/4CAP；checker按当前padded读取。若实际kernel ABI需要变动，应在统一GPU freeze前显式适配，不修改数值门槛。

## 尚未通过的阶段

1. `activation_shared_dsl.py` 只是新DSL翻译的CPU/AOT，不继承原C++ helper的2063exact。
2. 主composite kernel需完成全CTA/cluster publish、远端读完后复用、TMEM/SMEM资源组合以及实际cluster4 launch审计。
3. 当前harness callable尚未对新kernel运行；构造、graph lifetime、native finalizer binding将在统一GPU gate逐项验证。
4. debug boundary写回增加成本；完整debug链通过后，非debug输出一致性和计时应另行冻结，不能拿组件通过或资源合格声称服务收益。

`protocol-template.json` 明确标注TEMPLATE_CPU_ONLY，GPU、CUBIN、命令均留空；不能把模板当作已冻结实验。
