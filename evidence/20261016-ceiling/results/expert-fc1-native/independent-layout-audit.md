# FC1 exact FAIL 的独立 CPU 布局审计

2026-10-16；只读 runtime、实现者文件与原始失败数据，未运行GPU。只新增本报告及两个CPU诊断驱动/结果。原 `result.json` 的 `NATIVE_MATH_FAIL`、`timings_allowed=false` 不变。

## 结论

**4230/8192 FP8 byte差异是确定的每16字节输出排列不兼容，不是gate/up整组颠倒、signed zero或E4M3解码误差。Native oracle的逻辑排列有独立FP64矩阵计算支持；不能只改validator的reshape然后将候选判为可替代native。**

固定逆置换后FP8 bytes全部exact，但**35/64 FP32 scales仍不exact**。源码/本机native SASS显示两者dequant scale计算合同不同；必须修实现并重新走原gate，不能覆盖旧FAIL或直接放宽阈值。本轮没有性能结论。

## 1. 保存数据与独立复核

输入：`actual-reference.pt` SHA256 `907a14a8b0a9e61c3c80664f8d6849725ad9b92454ddde88a2b5159385b2e487`；`inputs.pt` SHA256 `7c5e0570c8c3df63bb5636c645b5bf3e71c000f6ef17f25f72710991e8d338f7`。原result SHA256 `4b1fbcb36eb56ed72862039e8ab7981e8ade1a2375fead891062c738e79b2299`。

固定layer0/expert205，8条真实route，token `[0,1,3,23,24,47,57,62]`、expanded `[2,11,30,186,193,378,457,499]`。candidate/raw reference均为CPU连续uint8 `[8,1024]`；scale是FP32 `[block128,token]=[8,8]`。下述变换不跨token、128量化block或512的up/gate边界。

```python
aligned = output_bytes.reshape(8, 64, 8, 2).transpose(-1, -2).reshape(8, 1024)
```

即candidate每16byte为native的 `[b0,b8,b1,b9,...,b7,b15]`。独立诊断得到：

| 检查 | 结果 |
|---|---:|
| raw byte不等 | 4230 / 8192 |
| 固定逆置换后不等 | **0 / 8192** |
| 各token整行byte multiset | 全部exact |
| signed-zero差异 | 0 |
| NaN | 两臂均0 |
| zero值 | 两臂均5184 |
| scale不等 | 35 / 64 |
| scale signed ULP(candidate−native) | −1:4，0:29，+1:30，+2:1 |
| scale max absolute / RRMS | 9.313225746e−10 / 6.760766535e−8 |
| 对齐bytes后dequant max absolute / RRMS | 4.768371582e−7 / 7.063926371e−8 |

原始`run.py:46–47`把两臂都按相同逻辑顺序解码，得dequant RRMS1.28794。这个数准确反映**当前不兼容输出直接交给相同consumer的错误**；它不是FC1实数GEMM算术误差的恰当归因。逆排列后的约7e−8差异则来自未解决的scale。

证据：`independent_cpu_layout.py` / `independent-layout-metrics.json`。

## 2. 哪一种才是正确逻辑排列

独立CPU参考只使用保存的E4M3输入/权重和FP32 block scales，未调用native/candidate kernel：对16个K128 block用FP64 GEMM，再乘对应输入及权重scale并累加。以保存的各臂output scales进行E4M3量化，仅用于判定逻辑row顺序，不是模拟native FP32归约/近似reciprocal。

- Native FC1 bytes与其FP64量化参考：8192/8192 exact。
- Candidate原始bytes与其FP64量化参考：4230不同。
- Candidate逆排列后bytes与其FP64量化参考：8192/8192 exact。
- Native/corrected candidate 的FP8反量化相对FP64 GEMM约2.403% RRMS，是既有E4M3量化误差；将up512/gate512交换后RRMS约1.405，故gate/up顺序不是这次故障。

为检查oracle提取不是一个偶然布局，读取 `expert-metadata-rebase` 的**全部15个scenario文件**（layers0/19/39 × real/left_only/right_only/rows_reversed/worst_global288），每个取一个明确存在的expert，分别加载对应checkpoint权重。共抽查78条route、79872bytes：native逻辑顺序与CPU FP64量化参考**0不同**；按candidate的16byte交织顺序解释则57035不同。不是全256专家全量验证；每case expert/行数/hash保存在`independent-oracle15-metrics.json`。

因此`expert-metadata-rebase/driver.py:99–105`按expanded→permuted映射提取native FC1行与scale的方案，有独立数学证据支持。本轮没有发现需要改oracle或validator才能解决的问题。若将来新的完整融合kernel选择内部不同布局，其后续activation必须同步理解该布局；当前目标是native-drop-in合同，必须修输出store或明确layout adapter。

## 3. 源码/编译产物支持的布局归因

installed `flashinfer/prims_ts/batched_gemm/gmem_c_resources.py`：

- `:2460–2463`说明native generated路径通过SMEM swizzle再TMA store，当前direct GMEM路径试图复现同一register grouping。
- `:2505–2513`在DeepSeek two-epilogue路径中，`base_tmem_col=(lane&3)*2`、`base_row_idx=warp*16+(lane>>2)*2`。
- `:3299–3339`取每lane的val0/val1，在`m_row0`、`m_row0+1`处打包并直接存相邻FP8 bytes。
- `:1788–1791`明确NVVM f8x2把第二operand放低byte；函数反序传参正好使val0低byte/val1高byte，**不是简单把cvt两个参数对调的错误**。

实际native SASS `../expert-fc1-native-v2/native-fc1.sass:1524–1530`为分开的`STS.U8`，`:1572`执行`UTMASTG.4D`。它不是候选的连续`st.global.b16`路径。观测的logical映射相当于对每16行将direct行索引转换为：

```text
logical_m = (m & ~15) + ((m & 15) >> 1) + ((m & 1) << 3)
```

即一对val0/val1实际属于同16行组内`j`与`j+8`，不是`2j`与`2j+1`。这是源级store布局差异与独立数学验证共同支持的修正方向。完整native STS地址表达式/TMA descriptor逐指令推导仍由实现者收尾；本报告不把单条SASS指令当所有shape映射的证明。修改必须限定实际触发的DeepSeek/two-epilogue分支，不能无条件改变其它FP8/gated/TMA路径。

## 4. scale是真实未解决的算术合同差异

Prims-TS `gmem_c_resources.py:1659–1662`：

```text
q  = 448 * approximate_rcp(safe_amax)
dq = amax * float32(1/448)
```

同文件`:1628–1631`另一路同样如此。冻结candidate PTX确有`rcp.approx.ftz`用于q，而dq用`mul.f32 ...,0f3B124925`。

已独立读取实现者从准确installed native cubin提取的SASS：

- `native-fc1.sass:1462` `MUFU.RCP`；`:1464`乘448得到q；`:1466`再次`MUFU.RCP`计算dq；`:1472`将dq写出。
- 第二个scale在`:1480–1496`重复同合同。
- SASS文件SHA256 `b3ff29aa91586ea8419922b6c31cedbc397fd68b62cf578822e5d25c3fa88c75`。

所以native **dq=approximate_rcp(q)**，而不是直接amax/448。两者在实数算术上等价，在FP32近似运算上不等价。观察到的±1/2ULP正符合这类合同差异，但本报告不声称“换公式已证明足以修复所有35个值”：仍需同输入GPU exact gate，尤其amax/乘法FTZ和零block下限也须保留native语义。CPU FP64不是近似MUFU的bit-level模拟器。

## 5. 后续判定边界

允许在artifact中按source-backed映射修store，并将dq计算恢复native reciprocal合同；不能直接改原result，也不能把8192byte逆置换后PASS替代输出合同gate。修正后仍先验证同8真实route的**原始输出bytes和64个scale全部exact**，再验证图重放、其它layer/expert/占用shape、unused/input/RNG ownership，全部通过才可谈局部timing。原失败保持可见，本CPU审计不授予性能或质量晋级。
