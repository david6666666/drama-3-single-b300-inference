"""Drama 3 execution with persistent slots and operator-only GPU dependencies.

Fish Speech supplies checkpoint containers and the prompt contract. No vLLM or
SGLang engine, scheduler, model runner, or Omni pipeline is instantiated here.
"""

import os

import flashinfer
import torch
import torch.nn.functional as F
from fish_speech.models.text2semantic.fish_omni.fp8_moe import quant_kernel, quantize_groups
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from flashinfer.fused_moe import WeightLayout, trtllm_fp8_block_scale_moe
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

from drama3.gdn import decode as gdn_decode
from drama3.gdn import prefill as gdn_prefill
from drama3.kernels import (
    add_norm,
    attention_metadata,
    combine_experts,
    conv_update,
    gated_norm,
    linear,
    native_moe,
    prepare_attention,
    route,
    silu_mul,
    small_batch_moe,
    table_projection,
)
from drama3.kernels import (
    sample_partitioned as sample,
)
from drama3.prefill_conv import prefill_conv


def blocked_moe_weight(weight):
    """Shuffle unchanged FP8 bytes into TRT-LLM's BlockMajorK layout.

    The batched row-gather follows vLLM's Apache-2.0 flashinfer_utils.py;
    permutation indices come from FlashInfer. Scales retain their logical order.
    """
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    experts, rows, columns = weight.shape
    order = get_shuffle_matrix_a_row_indices(weight[0].view(torch.uint8), 64).to(weight.device)
    blocks = weight.view(torch.uint8).view(experts, rows, columns // 128, 128)
    return blocks.permute(0, 2, 1, 3)[:, :, order, :].contiguous().view(weight.dtype)


class AttentionBatch:
    def __init__(
        self,
        slots: torch.Tensor,
        positions: torch.Tensor,
        max_length: int,
        backend: str = "cute-dsl",
        tensor_cores: bool = True,
    ):
        self.slots = slots
        self.positions = positions
        size = len(slots)
        self.page_size = 16
        if max_length % self.page_size:
            raise ValueError("Context capacity must be a multiple of 16")
        self.pages = max_length // self.page_size
        self.lengths = torch.full((size,), self.page_size, device="cuda", dtype=torch.int32)
        self.indptr = torch.arange(size + 1, device="cuda", dtype=torch.int32) * self.pages
        self.page_indices = (
            slots.to(torch.int32)[:, None] * self.pages
            + torch.arange(self.pages, device="cuda", dtype=torch.int32)
        ).flatten()
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.direct = backend == "cute-dsl"
        if self.direct:
            from flashinfer.cute_dsl.attention import BatchDecodePagedCuteDSLWrapper

            self.refresh()
            self.wrapper = BatchDecodePagedCuteDSLWrapper(self.workspace, use_cuda_graph=True)
            self.wrapper.plan(
                self.indptr,
                self.page_indices,
                self.lengths,
                num_qo_heads=16,
                num_kv_heads=2,
                head_dim=256,
                page_size=self.page_size,
                max_kv_len=max_length,
                reduction="kernel",
            )
            return
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            backend=backend,
            use_tensor_cores=tensor_cores,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.indptr,
            paged_kv_indices_buffer=self.page_indices,
            paged_kv_last_page_len_buffer=self.lengths,
        )
        self.plan(slots.cpu().tolist(), positions.cpu().tolist())

    def plan(self, slots: list[int], positions: list[int]):
        if self.direct:
            self.slots.copy_(torch.tensor(slots, dtype=torch.int64))
            self.positions.copy_(torch.tensor(positions, dtype=torch.int64))
            self.refresh()
            return
        counts = [(position + self.page_size) // self.page_size for position in positions]
        indptr = [0]
        indices = []
        for slot, count in zip(slots, counts, strict=True):
            indices.extend(range(slot * self.pages, slot * self.pages + count))
            indptr.append(len(indices))
        self.wrapper.plan(
            torch.tensor(indptr, dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor(
                [position % self.page_size + 1 for position in positions], dtype=torch.int32
            ),
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=256,
            page_size=self.page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    def run(self, queries, keys, values):
        shape = (-1, self.page_size, 2, 256)
        if self.direct:
            return self.wrapper.run(queries, keys.view(shape), values.view(shape), enable_pdl=False)
        return self.wrapper.run(queries, (keys.view(shape), values.view(shape)), enable_pdl=False)

    def refresh(self):
        attention_metadata(self.slots, self.positions, self.page_indices, self.lengths, self.pages)


class DramaModel:
    def __init__(self, checkpoint: str, slots: int = 64, max_length: int = 2048):
        if os.environ.get("DRAMA3_EXPERIMENTAL_DENSE_FP8") == "1":
            raise ValueError("Dense FP8 serving was rejected; only diagnostic scripts retain it")
        self.state_precision = os.environ.get("DRAMA3_GDN_STATE_DTYPE", "float32")
        if self.state_precision not in ("float32", "bfloat16"):
            raise ValueError("DRAMA3_GDN_STATE_DTYPE must be float32 or bfloat16")
        state_dtype = getattr(torch, self.state_precision)
        self.moe_layout = os.environ.get("DRAMA3_MOE_LAYOUT", "major_k")
        self.prefill_moe = os.environ.get("DRAMA3_PREFILL_MOE", "native")
        self.packed_moe_projection = os.environ.get("DRAMA3_PACKED_MOE_PROJECTION") == "1"
        self.prefill_conv_backend = os.environ.get("DRAMA3_PREFILL_CONV", "torch")
        if self.prefill_conv_backend not in ("torch", "triton"):
            raise ValueError("DRAMA3_PREFILL_CONV must be torch or triton")
        if self.moe_layout not in ("major_k", "blocked"):
            raise ValueError("DRAMA3_MOE_LAYOUT must be major_k or blocked")
        if self.prefill_moe not in ("native", "flashinfer"):
            raise ValueError("DRAMA3_PREFILL_MOE must be native or flashinfer")
        self.weights = FishOmniModel.from_pretrained(
            checkpoint,
            devices=["cuda:0"],
            tower_device="cuda:0",
            dtype=torch.bfloat16,
        )
        self.config = self.weights.config
        self.slots = slots
        self.max_length = max_length
        self.backbone = self.weights.backbone.model
        self.tower = self.weights.audio_decoder
        self.kv = {}
        self.states = {}
        for index, layer in enumerate(self.backbone.layers):
            # FlashInfer SwiGLU consumes [up | gate]; the checkpoint stores [gate | up].
            experts = layer.mlp.experts
            gate, up = experts.w13_weight.chunk(2, dim=1)
            experts.flashinfer_w13 = torch.cat((up, gate), dim=1)
            gate_scale, up_scale = experts.w13_scale_inv.chunk(2, dim=1)
            experts.flashinfer_w13_scale = torch.cat((up_scale, gate_scale), dim=1)
            experts.flashinfer_w2 = experts.w2_weight
            experts.flashinfer_shuffled = self.moe_layout == "blocked"
            if experts.flashinfer_shuffled:
                experts.flashinfer_w13 = blocked_moe_weight(experts.flashinfer_w13)
                experts.flashinfer_w2 = blocked_moe_weight(experts.w2_weight)
            if layer.layer_type == "full_attention":
                self.allocate_kv(index)
                attn = layer.self_attn
                attn.packed_qkv = torch.cat(
                    (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0
                )
            else:
                attn = layer.linear_attn
                attn.A_log_fp32 = attn.A_log.float()
                attn.dt_bias_fp32 = attn.dt_bias.float()
                attn.packed_input = torch.cat(
                    (
                        attn.in_proj_qkv.weight,
                        attn.in_proj_z.weight,
                        attn.in_proj_b.weight,
                        attn.in_proj_a.weight,
                    ),
                    dim=0,
                )
                self.states[index] = (
                    torch.zeros((slots, 8192, 3), dtype=torch.bfloat16, device="cuda"),
                    torch.zeros((slots, 32, 128, 128), dtype=state_dtype, device="cuda"),
                )
            layer.mlp.shared_expert.packed_gate_up = torch.cat(
                (
                    layer.mlp.shared_expert.gate_proj.weight,
                    layer.mlp.shared_expert.up_proj.weight,
                )
            )
            layer.mlp.packed_input_projection = F.pad(
                torch.cat(
                    (
                        layer.mlp.shared_expert.packed_gate_up,
                        layer.mlp.gate.weight,
                        layer.mlp.shared_expert_gate.weight,
                    )
                ),
                (0, 0, 0, 127),
            )
        for index, layer in enumerate([*self.tower.shared_layers, *self.tower.residual_layers]):
            self.allocate_kv(40 + index)
            mlp = layer.transformer_layer.feed_forward
            mlp.packed_gate_up = torch.cat((mlp.w1.weight, mlp.w3.weight))
            if index:
                embedding = layer.semantic_embedding if layer.shared else layer.codebook_embedding
                normed = self.norm(embedding.weight, layer.enorm)
                layer.code_table = (
                    normed.double() @ layer.eh_proj.weight[:, :2048].double().T
                ).float()
                layer.hidden_projection = layer.eh_proj.weight[:, 2048:].contiguous()
        self.candidates = (
            torch.cat(
                (
                    torch.tensor([self.config.eos_token_id], device="cuda"),
                    torch.arange(
                        self.config.semantic_start_token_id,
                        self.config.semantic_end_token_id + 1,
                        device="cuda",
                    ),
                )
            )
            .sort()
            .values
        )
        # Keep the semantic head aligned for tensor-core GEMMs; padded logits
        # are discarded before sampling, so the candidate vocabulary is intact.
        head = self.weights.backbone.lm_head.weight[self.candidates].contiguous()
        self.lm_head = F.pad(head, (0, 0, 0, (-len(head)) % 128))
        frequency = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
        angles = torch.arange(max_length, device="cuda").float()[:, None] * frequency[None, :]
        self.cosine = angles.cos()
        self.sine = angles.sin()
        self.sampling_step = torch.zeros((), dtype=torch.int64, device="cuda")
        self.dense_precision = "bfloat16"
        tune_cache = os.environ.get("DRAMA3_AUTOTUNE_CACHE")
        if tune_cache:
            with flashinfer.autotune(
                tune_mode=True,
                cache=tune_cache,
                tuning_buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
            ):
                dummy = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
                self.moe(dummy, self.backbone.layers[0].mlp, decode=True)

    def allocate_kv(self, index: int) -> None:
        shape = (self.slots, self.max_length, 2, 256)
        self.kv[index] = (
            torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
        )

    @staticmethod
    def norm(x: torch.Tensor, layer) -> torch.Tensor:
        return flashinfer.gemma_rmsnorm(x.contiguous(), layer.weight, layer.eps)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor, interleaved: bool) -> torch.Tensor:
        cosine, sine = self.cosine[positions, None], self.sine[positions, None]
        part = x[..., :64].float()
        if interleaved:
            first, second = part[..., ::2], part[..., 1::2]
            rotated = torch.stack(
                (first * cosine - second * sine, second * cosine + first * sine), dim=-1
            ).flatten(-2)
        else:
            first, second = part.chunk(2, dim=-1)
            rotated = torch.cat(
                (first * cosine - second * sine, second * cosine + first * sine), dim=-1
            )
        return torch.cat((rotated.to(x.dtype), x[..., 64:]), dim=-1)

    def attention(
        self,
        x: torch.Tensor,
        layer,
        index: int,
        positions: torch.Tensor,
        slots: torch.Tensor,
        batch: AttentionBatch | None,
        prefill_sequences: int = 1,
    ) -> torch.Tensor:
        tower = index >= 40
        projected = linear(x, layer.wqkv.weight if tower else layer.packed_qkv)
        keys, values = self.kv[index]
        if batch is not None:
            queries = prepare_attention(
                projected,
                layer.q_norm.weight,
                layer.k_norm.weight,
                self.cosine,
                self.sine,
                positions,
                slots,
                keys,
                values,
                layer.q_norm.eps,
                tower,
            )
            gate = (
                projected[:, 4096:8192]
                if tower
                else projected[:, :8192].reshape(-1, 16, 512)[:, :, 256:].reshape(-1, 4096)
            )
            out = batch.run(queries, keys, values)
            return linear(
                out.reshape(-1, 4096) * gate.sigmoid(),
                layer.wo.weight if tower else layer.o_proj.weight,
            )
        if tower:
            q, gate, k, v = projected.split((4096, 4096, 512, 512), dim=-1)
        else:
            qgate, k, v = projected.split((8192, 512, 512), dim=-1)
            q, gate = qgate.view(-1, 16, 512).chunk(2, dim=-1)
            gate = gate.reshape(-1, 4096)
        q = self.rotate(
            self.norm(q.reshape(-1, 256), layer.q_norm).reshape(-1, 16, 256), positions, tower
        )
        k = self.rotate(
            self.norm(k.reshape(-1, 256), layer.k_norm).reshape(-1, 2, 256), positions, tower
        )
        v = v.reshape(-1, 2, 256)
        keys[slots, positions] = k
        values[slots, positions] = v
        out = F.scaled_dot_product_attention(
            q.reshape(prefill_sequences, -1, 16, 256).transpose(1, 2),
            k.reshape(prefill_sequences, -1, 2, 256).transpose(1, 2),
            v.reshape(prefill_sequences, -1, 2, 256).transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        return linear(
            out.reshape(-1, 4096) * gate.sigmoid(),
            layer.wo.weight if tower else layer.o_proj.weight,
        )

    def linear_attention(
        self,
        x: torch.Tensor,
        layer,
        index: int,
        slots: torch.Tensor,
        decode: bool,
        valid_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = linear(x, layer.packed_input)
        mixed, z, b, a = projected.split((8192, 4096, 32, 32), dim=-1)
        conv, state = self.states[index]
        if decode:
            mixed = conv_update(mixed, conv, layer.conv1d.weight, slots)
            out = torch.empty((len(slots), 1, 32, 128), dtype=x.dtype, device=x.device)
            gdn_decode(mixed, a, b, layer.A_log_fp32, layer.dt_bias_fp32, state, out, slots)
        else:
            sequences = len(valid_length) if valid_length is not None else 1
            span = len(x) // sequences
            sequence_slots = slots[::span]
            if self.prefill_conv_backend == "triton":
                mixed = prefill_conv(
                    mixed, layer.conv1d.weight, conv, sequence_slots, span, valid_length
                )
            else:
                sequence = mixed.reshape(sequences, span, 8192).transpose(1, 2).contiguous()
                if valid_length is None:
                    conv[slots[:1]] = F.pad(sequence, (max(0, 3 - x.shape[0]), 0))[..., -3:]
                else:
                    indices = valid_length[:, None] - 3 + torch.arange(3, device=x.device)
                    conv[sequence_slots] = sequence.gather(
                        -1, indices[:, None, :].expand(-1, 8192, -1)
                    )
                mixed = F.silu(F.conv1d(sequence, layer.conv1d.weight, padding=3, groups=8192))
                mixed = mixed[..., :span].transpose(1, 2).reshape(-1, 8192).contiguous()
            q, k, v = mixed.split((2048, 2048, 4096), dim=-1)
            q = q.reshape(sequences, span, 16, 128)
            k = k.reshape(sequences, span, 16, 128)
            v = v.reshape(sequences, span, 32, 128)
            decay = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias.float())
            beta = b.float().sigmoid()
            if valid_length is not None:
                out = gdn_prefill(
                    q,
                    k,
                    v,
                    decay.reshape(sequences, span, 32),
                    beta.reshape(sequences, span, 32),
                    state,
                    slots,
                    valid_length,
                )
            else:
                decay = decay.reshape(sequences, span, 32)
                beta = beta.reshape(sequences, span, 32)
                out, _, _ = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    initial_state=state,
                    initial_state_indices=sequence_slots,
                    use_qk_l2norm_in_kernel=True,
                    inplace_update=True,
                )
        out = gated_norm(
            out.reshape(-1, 1, 32, 128), z.contiguous(), layer.norm.weight, layer.norm.eps
        ).reshape(-1, 4096)
        return linear(out, layer.out_proj.weight)

    def moe(self, x: torch.Tensor, layer, decode: bool = False) -> torch.Tensor:
        experts = layer.experts
        if self.packed_moe_projection:
            projected = linear(x, layer.packed_input_projection)
            gate_up, router, shared_gate = (
                projected[:, :1024],
                projected[:, 1024:1280],
                projected[:, 1280:1281],
            )
        else:
            gate_up = linear(x, layer.shared_expert.packed_gate_up)
            router = linear(x, layer.gate.weight)
            shared_gate = linear(x, layer.shared_expert_gate.weight)
        if len(x) <= 8 or (not decode and self.prefill_moe == "native"):
            if len(x) <= 8:
                weights, indices = route(router)
            else:
                probabilities = router.float().softmax(-1)
                weights, indices = probabilities.topk(8, dim=-1)
                weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
            if len(x) <= 8:
                routed = small_batch_moe(x, indices, weights, experts)
            else:
                routed = native_moe(x, indices, weights, experts)
        else:
            quantized, scales = quantize_groups(quant_kernel, x, x.shape[-1])
            routed = trtllm_fp8_block_scale_moe(
                routing_logits=router.float(),
                routing_bias=None,
                hidden_states=quantized,
                hidden_states_scale=scales.T.contiguous(),
                gemm1_weights=experts.flashinfer_w13,
                gemm1_weights_scale=experts.flashinfer_w13_scale,
                gemm2_weights=experts.flashinfer_w2,
                gemm2_weights_scale=experts.w2_scale_inv,
                num_experts=256,
                top_k=8,
                n_group=None,
                topk_group=None,
                intermediate_size=512,
                local_expert_offset=0,
                local_num_experts=256,
                routed_scaling_factor=1.0,
                routing_method_type=4,
                use_shuffled_weight=experts.flashinfer_shuffled,
                weight_layout=(
                    WeightLayout.BlockMajorK if experts.flashinfer_shuffled else WeightLayout.MajorK
                ),
                tune_max_num_tokens=4096,
            )
            if isinstance(routed, list):
                routed = routed[0]
        shared = linear(
            silu_mul(gate_up),
            layer.shared_expert.down_proj.weight,
        )
        return combine_experts(routed, shared, shared_gate)

    def backbone_forward(
        self,
        frames: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        batch: AttentionBatch | None,
        valid_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.tower.embed_backbone(frames, self.backbone.embed_tokens(frames[:, 0]))
        pending = None
        for index, layer in enumerate(self.backbone.layers):
            normalized = (
                self.norm(hidden, layer.input_layernorm)
                if pending is None
                else add_norm(
                    pending, hidden, layer.input_layernorm.weight, layer.input_layernorm.eps
                )
            )
            if layer.layer_type == "full_attention":
                mixed = self.attention(
                    normalized,
                    layer.self_attn,
                    index,
                    positions,
                    slots,
                    batch,
                    len(valid_length) if valid_length is not None else 1,
                )
            else:
                mixed = self.linear_attention(
                    normalized, layer.linear_attn, index, slots, batch is not None, valid_length
                )
            normalized = add_norm(
                mixed,
                hidden,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            pending = self.moe(normalized, layer.mlp, batch is not None)
        return add_norm(pending, hidden, self.backbone.norm.weight, self.backbone.norm.eps)

    def tower_layer(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
        semantic: torch.Tensor,
        extra: torch.Tensor | None,
        layer,
        index: int,
        positions: torch.Tensor,
        slots: torch.Tensor,
        batch: AttentionBatch | None,
        prefill_sequences: int = 1,
    ) -> torch.Tensor:
        if index:
            hidden = table_projection(hidden, layer, codes, semantic)
        else:
            embedding = layer.semantic_embedding if layer.shared else layer.codebook_embedding
            embedded = torch.where(semantic[:, None], embedding(codes.clamp(0, 4095)), 0.0)
            if extra is not None:
                embedded = embedded + extra
            hidden = linear(
                torch.cat(
                    (self.norm(embedded, layer.enorm), self.norm(hidden, layer.hnorm)), dim=-1
                ),
                layer.eh_proj.weight,
            )
        transformer = layer.transformer_layer
        mixed = self.attention(
            self.norm(hidden, transformer.attention_norm),
            transformer.attention,
            40 + index,
            positions,
            slots,
            batch,
            prefill_sequences,
        )
        normalized = add_norm(mixed, hidden, transformer.ffn_norm.weight, transformer.ffn_norm.eps)
        mixed = linear(
            silu_mul(linear(normalized, transformer.feed_forward.packed_gate_up)),
            transformer.feed_forward.w2.weight,
        )
        return add_norm(mixed, hidden, layer.final_layernorm.weight, layer.final_layernorm.eps)

    def tower_forward(
        self,
        hidden: torch.Tensor,
        frames: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        batch: AttentionBatch | None,
        labels: torch.Tensor | None,
        history: torch.Tensor | None = None,
        prefill_sequences: int = 1,
    ) -> torch.Tensor:
        if labels is None:
            self.sampling_step.add_(1)
            logits = linear(hidden, self.lm_head)[:, : len(self.candidates)]
            first = self.candidates[sample(logits, self.sampling_step, 0)]
            retry = self.candidates[sample(logits, self.sampling_step, 1, top_p=0.95)]
            repeated = (history == first[:, None]).any(-1) & self.tower.is_semantic(first)
            main = torch.where(repeated, retry, first)
            code = (main - self.tower.semantic_begin).clamp(0, 4095)
        else:
            main, code = labels[:, 0], labels[:, 1]
        semantic = self.tower.is_semantic(main)
        extra = self.tower.embedding_sum(frames, 2, 10)
        extra = torch.where(self.tower.is_semantic(frames[:, 0])[:, None], extra, 0.0)
        for index, layer in enumerate(self.tower.shared_layers):
            hidden = self.tower_layer(
                hidden,
                code,
                semantic,
                extra if index == 0 else None,
                layer,
                index,
                positions,
                slots,
                batch,
                prefill_sequences,
            )
        codes = [code]
        for index, (layer, head) in enumerate(
            zip(self.tower.residual_layers, self.tower.codebook_heads, strict=True)
        ):
            hidden = self.tower_layer(
                hidden,
                codes[-1] if labels is None else labels[:, index + 1],
                semantic,
                None,
                layer,
                4 + index,
                positions,
                slots,
                batch,
                prefill_sequences,
            )
            if labels is None:
                codes.append(sample(linear(hidden, head.weight), self.sampling_step, index + 2))
        return torch.stack((main, *codes), dim=-1)

    @torch.inference_mode()
    def prefill_batched(self, frames, request_slots, lengths):
        sequences, span, _ = frames.shape
        for conv, state in self.states.values():
            conv.index_fill_(0, request_slots, 0)
            state.index_fill_(0, request_slots, 0)
        positions = torch.arange(span, device=frames.device).repeat(sequences)
        slots = request_slots[:, None].expand(-1, span).reshape(-1)
        hidden = self.backbone_forward(frames.reshape(-1, 11), positions, slots, None, lengths)
        hidden = hidden.reshape(sequences, span, -1)
        self.tower_forward(
            hidden[:, :-1].reshape(-1, 2048),
            frames[:, :-1].reshape(-1, 11),
            positions.reshape(sequences, span)[:, :-1].reshape(-1),
            slots.reshape(sequences, span)[:, :-1].reshape(-1),
            None,
            frames[:, 1:].reshape(-1, 11),
            prefill_sequences=sequences,
        )
        return hidden[torch.arange(sequences, device=frames.device), lengths - 1]

    @torch.inference_mode()
    def prefill_padded(
        self,
        frames: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        for conv, state in self.states.values():
            conv.index_fill_(0, slots[:1], 0)
            state.index_fill_(0, slots[:1], 0)
        hidden = self.backbone_forward(frames, positions, slots, None, length)
        self.tower_forward(hidden[:-1], frames[:-1], positions[:-1], slots[:-1], None, frames[1:])
        return hidden.index_select(0, length - 1)

    @torch.inference_mode()
    def prefill(self, frames: torch.Tensor, slot: int) -> torch.Tensor:
        if len(frames) > self.max_length:
            raise ValueError("Prompt exceeds the configured context length")
        for conv, state in self.states.values():
            conv[slot].zero_()
            state[slot].zero_()
        frames = frames.to("cuda")
        positions = torch.arange(len(frames), device="cuda")
        slots = torch.full_like(positions, slot)
        hidden = self.backbone_forward(frames, positions, slots, None)
        if len(frames) > 1:
            self.tower_forward(
                hidden[:-1], frames[:-1], positions[:-1], slots[:-1], None, frames[1:]
            )
        return hidden[-1:]

    @torch.inference_mode()
    def decode(
        self,
        frames: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        history: torch.Tensor,
        batch: AttentionBatch,
    ) -> torch.Tensor:
        hidden = self.backbone_forward(frames, positions, slots, batch)
        return self.tower_forward(hidden, frames, positions, slots, batch, None, history)
