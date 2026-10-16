# Codec transformer 参数转换：CPU 合同审计

结论：该模型的 **40 个 transformer dense weight 的 FP32→BF16 转换可以作为只执行一次的候选**，但必须保留原 FP32 参数、当前 norm/layer-scale/残差行为以及所有流状态。推荐首先试验 `torch.compile(..., options={"freezing": True, "freezing_discard_parameters": False, "enable_linear_binary_folding": False})`，因为它无需修改共享 codec 的参数 dtype；这是带有更广编译优化的实验，不能未经 exact gate 就称为“仅消除复制”。CPU 审计没有运行 GPU、修改 runtime 或创建候选编译图。

## 实际参数集合与所有权

冻结 runtime source=`69d636ef11875add6e4edf7c9bb028b9574143a3`。CPU `torch.load(codec.pth, mmap=True, weights_only=True, map_location="cpu")` 核对了 checkpoint 元数据，结束时 `torch.cuda.is_initialized()==False`。模型是 `/home/zjy/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/e0a9218aa48888270f06ef1ed9ea67622c0e17d2`。

只收集 `codec.quantizer.post_module.layers[i]`，i=0..7，以下五个标准 `nn.Linear.weight`：

| 相对名称 | shape | dtype | bias |
| --- | --- | --- | --- |
| attention.wqkv.weight | 3072×1024 | FP32 | None |
| attention.wo.weight | 1024×1024 | FP32 | None |
| feed_forward.w1.weight | 3072×1024 | FP32 | None |
| feed_forward.w3.weight | 3072×1024 | FP32 | None |
| feed_forward.w2.weight | 1024×3072 | FP32 | None |

合计40个矩阵、109,051,904个元素；原FP32为416MiB，对应BF16缓存为208MiB。input_proj/output_proj 都是 Identity，look_ahead_conv 是 Identity，没有额外 projection bias。实际 checkpoint 没有 decoder transformer 参数；上游 DecoderBlock 虽构造了 transformer_module，但没有把它注册进 block（`modded_dac.py:740`附近的注释）。不要从YAML的 decoder_transformer_layers推断还应转换另一组权重。

`ChannelsLastDACDecoder` 把 `self.transformer` 直接指向 `quantizer.post_module`（冻结 `ops/dac_channels_last.py:967`），没有clone。checkpoint 中上述40个post weight与对应pre weight没有共享storage，但原post module仍由整个codec持有；`DAC.encode()`也会进入 `quantizer.forward()`，其后半段调用post_module，尽管输出codes在此之前已经生成。原FP32参数因此应完全保留，不能原地把共享post module的weight替换成BF16。

最小显式缓存备选是：decoder拥有只读BF16 tensor缓存，键为上述module身份/已验证名称；只在decoder的linear运算消费缓存，原module.weight/bias不变。cache在加载、bake_weight_norm、eval、device确定之后且compile/capture之前建立，使用与当前device转换相同的FP32→BF16 round。不得每次forward重新创建。若用独立module clone，应只替换clone内这40个linear参数；直接deepcopy整个transformer还会复制很大的causal_mask/frequency buffer，不是必要条件，应复用已确认只读buffer。没有依据扩展到全codec.to(bfloat16)。

## 不可改变的 dtype 语义

上游源码 `/home/zjy/code/hsliu/drama3-sources/fish-speech/fish_speech/models/dac/modded_dac.py`：

- `RMSNorm.forward:329` 先 `x.float()`求norm，再 `.type_as(x)`，最后乘 **FP32 norm.weight**。该乘法的类型提升及编译融合属于现有路径；把norm.weight转BF16会改变行为。所有17个norm weight必须保留FP32。
- `LayerScale.forward:345` 使用 `x.mul_(FP32 gamma)`，输出留在x dtype且有原位舍入语义；16个gamma不能预乘进dense weight，不能改成BF16，也不能改成out-of-place类型提升。
- `FeedForward.forward:316` 的w1→SiLU、w3分支、乘法、dropout、w2顺序保持；eval使dropout关闭。不能因共享input就自行合并GEMM并声称bitwise保持。
- `WindowLimitedTransformer.setup_caches`/父类的cache dtype取norm.weight.dtype（:131）；即便当前decoder使用外部ring，也不应修改此dtype作为副作用。
- runtime `_window_transformer:642` 按输入x.is_cuda/x.dtype和head形状选择ring/kv路径。预转换只能改变weight的供应位置，输入激活dtype、最终norm输出dtype、residual、rotary、KV ring与后续upsample均保持当前形式。
- encoder、quantizer.pre_module、embedding/code tables、ConvNeXt的dtype-sensitive分支、conv/upsample权重与bias缓存都不在本候选范围内。若未来支持FP16，需独立缓存/图；不能把BF16缓存用于其它autocast模式。

## 已存在编译缓存证据

CPU读取的BF16 ring图缓存：

`/tmp/torchinductor_zjy/7t/c7td5afmw4cbtftkfzfbuiwvthkwdazclkto4xf33uohfw4eaucr.py`

SHA256=`b6b07c4e78c23ae524308b66e61f75ef1cbb35dd267b1578bfa67d74ad4d30e7`。该文件只是现存相同结构的缓存样本，未将它冒认为当前profile唯一命中的cache key。

其 `triton_poi_fused__to_copy_1`(:144)签名是FP32输入/BF16输出，固定3,145,728元素，读取权重后直接store转换；`:926`对应wqkv，`:978`、`:987`对应两路FFN，w2同大小，合计32次。`_to_copy_3`(:382)是1,048,576元素的wo，8次。生成的matmul仍是BF16×BF16输出BF16，原参数以FP32输入placeholder进入图；shape/stride固定而batch/frame有符号维。没有导入或执行缓存文件。

因此root新profile summary中的 **40次cast=32大矩阵+8小矩阵** 与实际40个参数吻合。B4/F8这40次kernel合计141.275us，占kernel-duration sum的10.8901%；其中32次大矩阵为122.301us（9.4275%）。这是单图profile，不是服务墙钟或重叠窗口占比。每图完整cast的逻辑read+write为624MiB，但cache命中、图调度与其它kernel可能变化，不能按理论字节数推出端到端收益。若只删除同一profile里的所有40段cast、其它完全不变，纯kernel sum上限约1/(1−0.108901)=1.1222×；这不是测量结果。

## freezing机制与需要防住的扩大范围

安装PyTorch `torch/_inductor/config.py:1559`说明freezing会把权重当常量，之后不得更新；`:1565`默认 `freezing_discard_parameters=False`。本机CPU读到freezing=False、discard=False、enable_linear_binary_folding=False。编译入口 `compile_fx.py:2939` 还要求 `not torch.is_grad_enabled()`；decoder.__call__已有inference_mode，但artifact若直接调用内部函数必须显式保持inference_mode。

`freezing.py:29`的replace_params_with_constants跳过已识别的mutated/aliased参数，其余参数变成get_attr；非参数输入索引保留。`:127`执行freezing_passes和constant_fold。discard=False不会调用invalidate_eager_modules，不会删除原FP32参数。但常量折叠仍假定这些值以后不变，若reload权重或改device/dtype，必须重建编译图和CUDA graph，不能依靠旧freeze cache自动反映更新。

三个被compile的方法 `_transformer_layers`、`_transformer_layers_kv`、`_transformer_layers_ring` 可以各自使用选项。真正流状态rings/cached_k/v、starts、slots、lengths及freqs都是方法显式Tensor参数，不能作为闭包常量绑定。ring算子 `slot_attention.py:487` 声明 `mutates_args=("cache",)`，应继续保留为可变运行时输入；这是fullstate测试的重点。

freezing不仅fold casts：

- `fx_passes/freezing_patterns.py:37`运行binary_folding及其它patterns。`binary_folding.py:302`明确由 `config.enable_linear_binary_folding`控制linear的constant add/mul折叠，**显式pin False**可避免将FP32gamma折入BF16 GEMM权重。
- `freezing_patterns.py:166`的check_concat_weights在GPU不受 `cpp.enable_concat_linear=False`限制；`:243`二路matmul concat可能把w1/w3两GEMM合并。改变GEMM形状可能改变舍入，必须检查生成图和exact状态/PCM。
- norm/rotary等活跃输入的数学不能通过“参数constantfold”证明完全不变；常量化会影响融合、layout、kernel选择。需记录输出code、kernel signatures、mm形状、dtype、stride、autocast、torch/Triton版本及config完整值。

因此“无模块dtype修改”是原参数/encoder隔离上的优势；**数值原生等价仍由对照证明**，不是freezing选项本身保证。

## 有界artifact A/B应满足的合同

先同source69d636e、同FP32参数、BF16 autocast、private codec capture stream、同batch/frame、同初始完整state、同codes比较freeze0/1。编译/捕获在timing外，原40个weight及norm/gamma/buffer的bytes/dtype/storage在前后核对；inference tensor不保证提供_version，不能用_version访问失败代替验收。

首例要求 **PCM、全部真实slot和scratch/不参与slot状态、metadata完全一致**，记录首差异。原参数clone/hash完全一致只证明没有修改原模块；单例输出clone完全一致只证明该输入，不能证明所有动态slot/shape正确。通过后再固定DPcost表的80请求多shape/slotreuse/codec重叠重放，覆盖全部合法graph shape，确认graph引用的BF16常量生命周期固定、独立stream workspace保持。

同时profile确认40次cast实际被消除/减少，而非改名或迁移到timing外的请求路径。完整同源服务1088+全量WER/SIM才可晋级。若freezing触发不同GEMM导致exact失败，保留该失败，优先回退到显式40权重缓存的更窄候选，不得默认放宽gate。

## 随后完成的同源试验更新

Root 的 [codec-constant-fold/result.json](../codec-constant-fold/result.json) 已完成：273项exact检查通过，但状态为 **LOCAL_GAIN_FAIL**。B4/B8局部差异仅约0.17%/0.64%，两组仍各285个kernel并保留全部40个参数cast；因此本次bound-object compile上的 `freezing=True` 没有实现假设中的cast hoisting。不能将配置已启用等同于常量折叠发生。后续更窄的decoder-owned BF16 linear缓存属于独立候选；上述原参数、dtype及状态合同继续适用。
