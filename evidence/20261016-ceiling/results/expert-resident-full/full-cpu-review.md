# Full256 resident chain CPU 审查与验证准备

本轮已完成 full256 wrapper、固定地址metadata ABI、完整15oracle原生finalizer验证驱动、CPU AOT及资源审查。尚不把CPU结果当作整链GPU精度或性能结果；首次实际执行由协调agent统一冻结和启动。

## 原生坐标与完整cluster

outer固定grid `[4,288,1]`、cluster `[4,1,1]`；`blockIdx.y`是一组native token8 metadata tile。每个cluster的四个rank共同处理同一专家、同一个token8组。`WorkTileInfo=(rank+4*phase,tile,0)`保持原生A权重坐标和B路由语义，不把expert ID误当tile ID。

原生 `tile[tile]`给expert，`mn[tile]`是绝对结束行；有效行数是clamp(mn−8tile,0,8)，route[8tile+row]给原始token。一个专家超过8个route由多个独立metadata tile处理。未分配route的专家没有tile。outer统一用 `tile<ncta` 包住整个allocation/phase/DSM/deallocation生命周期，四rank同时进入或同时跳过；cluster只沿x扩展，ncta无需被4整除。固定grid允许同一CUDA graph更新ncta/padded，避免host读取future metadata。

单tile0版本的两个特定限制已在full fork显式修正：FC1 pinned shared的global n_col改为本地0…7（%8），避免后续tile越界；FC2 shared-B的 `mn_limit` 已由原生GmemB转换成本地count，因此不再第二次减 `coord_b_mn`。共享SF继承原生的local token limit，保持K128/expert/outputtile索引。

FC1 bytes的debug输出仍为global row×1024+原生up/gate列；activation为global row×512+rank128+channel。scale输出地址使用 **block×runtime_padded+global_row**，不是CAP stride；flat 8CAP/4CAP容量用于捕获可变padded布局。原生BF16 FC2 TMA输出与原finalizer不改排列。

`full-metadata-audit.json`枚举全部15归档oracle，并验证512expanded与native permutation的双射、expert匹配、本地row连续性、四rank完整列覆盖和scale边界。ncta从109到288，含inactive experts及多组专家；另枚举0/1/2/3/4/5/109/287/288的cluster一致性。这是CPU映射检查，不是执行生成的device code。

## 编译与资源

实际full CUBIN `09423f0fc6d503e71d781e1d76e6dfb07522ee1c6a8d97292e81aba6b41f57f0`，host请求78,464 B dynamic，加CUBIN1,024 B static，共79,488 B/CTA；STACK/LOCAL/PTX local/SASS spill均0。保留每阶段原M64N8K32/K128与160/48动态register角色。资源符合80KiB预算，但没有occupancy或速度推论。

编译器报告 `PHASE_AUTO_PROMOTED_TO_STAGED`。没有为消警告盲改数学或控制流；`full-phase-audit.json`验证实际生成PTX六个LoadA坐标为ctaid.x×128+[0,512,0,512,1024,1536]，四个FC2 TMA store保留[0,512,1024,1536]偏移及原生64列epilogue子块。未发现退化为phase0；GPU所有边界仍须独立验收。

## 执行接口与未覆盖项

`prepare_full.py`只在CUDA_VISIBLE_DEVICES为空时执行CPU/AOT；`run_full_gate.py`调用原 `validate_chain.run_full`，每层捕获一次graph并切换五种原生metadata；全15case逐字节检查FC1/activation/FC2、padding/tail、immutable、RNG和原native ordered-top8 finalizer。CPU/GPU CUBIN身份必须匹配；首次差异存档停止，不计时。三层完整原始权重通过safetensors按up、gate、down读取，FP8/scale不改。

`launch_full_gate.py`使用协议给定GPU UUID、watchdog和2秒采样进程记录；采样不称为持续排他证明。原生15case含大量inactive clusters，**不含ncta=0的整launch输入**；全空的CPU生命周期证明不能当作GPU全空测试。若full gate通过，需要另行声明该case后验证，不在此隐含放宽/扩大PASS。

四个新wrapper/driver文件最终Ruff通过。早先CPU映射辅助脚本首次把torch.where返回的expanded顺序当成permuted顺序，导致辅助断言失败；改为显式inverse permutation后通过。候选kernel、nativeoracle与GPU validator未因此改变，此辅助错误不计为候选精度失败或通过。
