"""Small fused operators for the fixed Drama 3 decode geometry."""

import torch
import triton
import triton.language as tl


@triton.jit
def conv_update_kernel(X, C, W, S, OUT, X_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(S + row)
    offset = (slot * 8192 + channels) * 3
    a = tl.load(C + offset).to(tl.float32)
    b = tl.load(C + offset + 1).to(tl.float32)
    c = tl.load(C + offset + 2).to(tl.float32)
    d = tl.load(X + row * X_STRIDE + channels).to(tl.float32)
    value = a * tl.load(W + channels * 4).to(tl.float32)
    value += b * tl.load(W + channels * 4 + 1).to(tl.float32)
    value += c * tl.load(W + channels * 4 + 2).to(tl.float32)
    value += d * tl.load(W + channels * 4 + 3).to(tl.float32)
    tl.store(C + offset, b)
    tl.store(C + offset + 1, c)
    tl.store(C + offset + 2, d)
    # Match the BF16 convolution output boundary before the SiLU activation.
    value = value.to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * 8192 + channels, value * tl.sigmoid(value))


def conv_update(
    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((len(slots), 8192), dtype=x.dtype, device=x.device)
    conv_update_kernel[(len(slots), 32)](x, state, weight, slots, output, x.stride(0), 256)
    return output


@triton.jit
def sample_kernel(
    L,
    STEP,
    OUT,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_P: tl.constexpr,
    STREAM: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    logits = tl.load(L + row * STRIDE + columns, columns < WIDTH, other=-float("inf")).to(
        tl.float32
    )
    maximum = tl.max(logits, 0)
    denominator = tl.sum(tl.exp(logits - maximum), 0)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    # Logits are BF16: their ordered FP32 keys have sixteen unused low bits.
    keys = (ordered & 0xFFFF0000) | (65535 - columns.to(tl.uint32))
    selected = tl.topk(keys, 32)
    ids = (65535 - (selected & 65535)).to(tl.int32)
    ranks = tl.arange(0, 32)
    values = tl.load(L + row * STRIDE + ids, ids < WIDTH, other=-float("inf")).to(tl.float32)
    probability = tl.exp(values - maximum) / denominator
    cumulative = tl.cumsum(probability, 0)
    kept = ((cumulative <= TOP_P) & (ranks < 30)) | (ranks == 0)
    probability = tl.where(kept, probability, 0.0)
    step = tl.load(STEP)
    random = tl.rand(SEED, (step * 64 * 16 + row * 16 + STREAM) * 32 + ranks)
    noise = -tl.log(tl.maximum(random, 1.0e-7))
    winner = tl.argmax(probability / noise, 0)
    choice = tl.sum(tl.where(ranks == winner, ids, 0), 0)
    tl.store(OUT + row, choice)


def sample(
    logits: torch.Tensor, step: torch.Tensor, stream: int, top_p: float = 0.9
) -> torch.Tensor:
    if logits.dtype != torch.bfloat16:
        raise TypeError("The fixed sampler expects BF16 logits")
    output = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    sample_kernel[(logits.shape[0],)](
        logits,
        step,
        output,
        logits.stride(0),
        logits.shape[1],
        top_p,
        stream,
        42,
        triton.next_power_of_2(logits.shape[1]),
        num_warps=8,
    )
    return output


@triton.jit
def attention_prepare_kernel(
    P,
    QW,
    KW,
    COS,
    SIN,
    POS,
    SLOT,
    QOUT,
    KC,
    VC,
    ROW_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    EPS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 256)
    position, slot = tl.load(POS + row), tl.load(SLOT + row)
    q_offset = head * (256 if INTERLEAVED else 512)
    raw = tl.load(P + row * ROW_STRIDE + q_offset + d).to(tl.float32)
    normalized = raw * tl.rsqrt(tl.sum(raw * raw, 0) / 256 + EPS)
    normalized *= 1.0 + tl.load(QW + d).to(tl.float32)
    normalized = normalized.to(P.dtype.element_ty).to(tl.float32)
    if INTERLEAVED:
        partner = d ^ 1
        frequency = d // 2
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
    else:
        partner = tl.where(d < 32, d + 32, d - 32)
        frequency = d % 32
        sign = tl.where(d < 32, -1.0, 1.0)
    peer = tl.gather(normalized, tl.minimum(partner, 255), axis=0)
    cosine = tl.load(COS + position * 32 + frequency, d < 64, other=1.0)
    sine = tl.load(SIN + position * 32 + frequency, d < 64, other=0.0)
    rotated = tl.where(d < 64, normalized * cosine + sign * peer * sine, normalized)
    tl.store(QOUT + (row * 16 + head) * 256 + d, rotated)
    if head < 2:
        raw = tl.load(P + row * ROW_STRIDE + 8192 + head * 256 + d).to(tl.float32)
        normalized = raw * tl.rsqrt(tl.sum(raw * raw, 0) / 256 + EPS)
        normalized *= 1.0 + tl.load(KW + d).to(tl.float32)
        normalized = normalized.to(P.dtype.element_ty).to(tl.float32)
        peer = tl.gather(normalized, tl.minimum(partner, 255), axis=0)
        rotated = tl.where(d < 64, normalized * cosine + sign * peer * sine, normalized)
        offset = ((slot * CAPACITY + position) * 2 + head) * 256 + d
        tl.store(KC + offset, rotated)
        value = tl.load(P + row * ROW_STRIDE + 8704 + head * 256 + d)
        tl.store(VC + offset, value)


def prepare_attention(
    projected,
    q_weight,
    k_weight,
    cosine,
    sine,
    positions,
    slots,
    keys,
    values,
    epsilon,
    interleaved,
):
    queries = torch.empty((len(positions), 16, 256), dtype=projected.dtype, device=projected.device)
    attention_prepare_kernel[(len(positions), 16)](
        projected,
        q_weight,
        k_weight,
        cosine,
        sine,
        positions,
        slots,
        queries,
        keys,
        values,
        projected.stride(0),
        keys.shape[1],
        epsilon,
        interleaved,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return queries


@triton.jit
def gated_norm_kernel(X, Z, W, OUT, EPS: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    x = tl.load(X + row * 128 + d).to(tl.float32)
    norm = x * tl.rsqrt(tl.sum(x * x, 0) / 128 + EPS)
    norm = norm.to(X.dtype.element_ty).to(tl.float32)
    scaled = (norm * tl.load(W + d).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    z = tl.load(Z + row * 128 + d).to(tl.float32)
    tl.store(OUT + row * 128 + d, scaled * (z * tl.sigmoid(z)))


def gated_norm(x, z, weight, epsilon):
    output = torch.empty_like(x)
    gated_norm_kernel[(x.numel() // 128,)](
        x, z, weight, output, epsilon, num_warps=4, enable_fp_fusion=False
    )
    return output


@triton.jit
def gemv_kernel(
    X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCK_K: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK_K)
    x = tl.load(X + columns, columns < K, other=0.0).to(tl.float32)
    weight = tl.load(
        W + rows[:, None] * K + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(weight * x[None, :], 1)
    tl.store(Y + rows, out, rows < N)


def gemv(x, weight, rows=4):
    output = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    gemv_kernel[(triton.cdiv(weight.shape[0], rows),)](
        x,
        weight,
        output,
        weight.shape[0],
        weight.shape[1],
        rows,
        triton.next_power_of_2(weight.shape[1]),
        num_warps=4 if weight.shape[1] <= 4096 else 8,
    )
    return output


def linear(x, weight):
    if x.shape[0] == 1 and x.dtype == torch.bfloat16:
        n, k = weight.shape
        rows = 1 if n <= 1024 else 2 if k == 4096 else 8 if n >= 12288 else 4
        return gemv(x, weight, rows)
    m, k = x.shape
    n = weight.shape[0]
    if 2 <= m <= 32 and (
        k >= 4096 or (k == 2048 and (256 <= n <= 1024 or (256 <= n <= 4096 and m <= 8)))
    ):
        from flashinfer.gemm import mm_bf16

        return mm_bf16(x.contiguous(), weight.T, backend="cute-dsl")
    return torch.nn.functional.linear(x, weight)


@triton.jit
def expert_up_kernel(X, XS, W, WS, IDS, OUT, ROWS: tl.constexpr):
    batch, expert_rank, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    expert = tl.load(IDS + batch * 8 + expert_rank)
    n = tile * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, 2048)
    x = tl.load(X + batch * 2048 + k).to(tl.float32)
    x_scale = tl.load(XS + batch * 16 + k // 128)
    w = tl.load(W + expert * 1024 * 2048 + n[:, None] * 2048 + k[None, :]).to(tl.float32)
    scale = tl.load(WS + expert * 8 * 16 + (n[:, None] // 128) * 16 + k[None, :] // 128)
    output = tl.sum(w * scale * (x * x_scale)[None, :], 1)
    tl.store(OUT + (batch * 8 + expert_rank) * 1024 + n, output)


@triton.jit
def expert_down_kernel(X, XS, W, WS, IDS, ROUTING, OUT, ROWS: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    rank = tl.arange(0, 8)
    experts = tl.load(IDS + batch * 8 + rank)
    n = tile * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, 512)
    x = tl.load(X + (batch * 8 + rank[:, None]) * 512 + k[None, :]).to(tl.float32)
    x_scale = tl.load(XS + (batch * 8 + rank[:, None]) * 4 + k[None, :] // 128)
    offset = experts[:, None, None] * 2048 * 512 + n[None, :, None] * 512 + k[None, None, :]
    w = tl.load(W + offset).to(tl.float32)
    scale_offset = (
        experts[:, None, None] * 16 * 4 + (n[None, :, None] // 128) * 4 + k[None, None, :] // 128
    )
    scale = tl.load(WS + scale_offset)
    expert_output = tl.sum(w * scale * (x * x_scale)[:, None, :], 2)
    weights = tl.load(ROUTING + batch * 8 + rank).to(tl.float32)
    output = tl.sum(expert_output * weights[:, None], 0)
    tl.store(OUT + batch * 2048 + n, output)


def small_batch_moe(x, topk_ids, topk_weights, experts):
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
        quant_kernel,
        quantize_groups,
        silu_mul_quant_kernel,
    )

    quantized, scales = quantize_groups(quant_kernel, x, 2048)
    gate_up = torch.empty((len(x) * 8, 1024), device=x.device, dtype=x.dtype)
    expert_up_kernel[(len(x), 8, 256)](
        quantized,
        scales,
        experts.w13_weight,
        experts.w13_scale_inv,
        topk_ids,
        gate_up,
        4,
        num_warps=4,
    )
    activated, activation_scales = quantize_groups(silu_mul_quant_kernel, gate_up, 512)
    output = torch.empty_like(x)
    expert_down_kernel[(len(x), 512)](
        activated,
        activation_scales,
        experts.w2_weight,
        experts.w2_scale_inv,
        topk_ids,
        topk_weights,
        output,
        4,
        num_warps=8,
    )
    return output


@triton.jit
def route_kernel(LOGITS, IDS, WEIGHTS):
    row = tl.program_id(0)
    columns = tl.arange(0, 256)
    logits = tl.load(LOGITS + row * 256 + columns).to(tl.float32)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    keys = (ordered & 0xFFFF0000) | (65535 - columns.to(tl.uint32))
    selected = tl.topk(keys, 8)
    ids = (65535 - (selected & 65535)).to(tl.int32)
    values = tl.load(LOGITS + row * 256 + ids).to(tl.float32)
    numerator = tl.exp(values - tl.max(logits, 0))
    probability = numerator / tl.sum(tl.exp(logits - tl.max(logits, 0)), 0)
    weights = probability / tl.sum(probability, 0)
    ranks = tl.arange(0, 8)
    tl.store(IDS + row * 8 + ranks, ids)
    tl.store(WEIGHTS + row * 8 + ranks, weights)


def route(logits):
    indices = torch.empty((len(logits), 8), dtype=torch.int64, device=logits.device)
    weights = torch.empty((len(logits), 8), dtype=logits.dtype, device=logits.device)
    route_kernel[(len(logits),)](logits, indices, weights, num_warps=4)
    return weights, indices


@triton.jit
def silu_mul_kernel(X, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * WIDTH * 2 + col, col < WIDTH, other=0).to(tl.float32)
    up = tl.load(X + row * WIDTH * 2 + WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * WIDTH + col, activated * up, col < WIDTH)


def silu_mul(projected):
    width = projected.shape[-1] // 2
    output = torch.empty((len(projected), width), dtype=projected.dtype, device=projected.device)
    silu_mul_kernel[(len(projected), triton.cdiv(width, 256))](projected, output, width, 256)
    return output


@triton.jit
def add_norm_kernel(X, RESIDUAL, W, OUT, EPS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    value = tl.load(X + row * WIDTH + col).to(tl.float32) + tl.load(
        RESIDUAL + row * WIDTH + col
    ).to(tl.float32)
    value = value.to(X.dtype.element_ty).to(tl.float32)
    tl.store(RESIDUAL + row * WIDTH + col, value)
    value *= tl.rsqrt(tl.sum(value * value, 0) / WIDTH + EPS)
    value *= 1.0 + tl.load(W + col).to(tl.float32)
    tl.store(OUT + row * WIDTH + col, value)


def add_norm(x, residual, weight, epsilon):
    output = torch.empty_like(x)
    add_norm_kernel[(len(x),)](
        x, residual, weight, output, epsilon, x.shape[-1], num_warps=4, enable_fp_fusion=False
    )
    return output


@triton.jit
def combine_experts_kernel(ROUTED, SHARED, GATE, OUT):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * 256 + tl.arange(0, 256)
    routed = tl.load(ROUTED + row * 2048 + col).to(tl.float32)
    shared = tl.load(SHARED + row * 2048 + col).to(tl.float32)
    gate = tl.sigmoid(tl.load(GATE + row).to(tl.float32)).to(SHARED.dtype.element_ty).to(tl.float32)
    weighted = (shared * gate).to(SHARED.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * 2048 + col, routed + weighted)


def combine_experts(routed, shared, gate):
    output = torch.empty_like(routed)
    combine_experts_kernel[(len(routed), 8)](routed, shared, gate, output)
    return output


@triton.jit
def sample_partition_kernel(
    L, KEYS, MAXIMUM, SUM, STRIDE: tl.constexpr, WIDTH: tl.constexpr, PARTS: tl.constexpr
):
    row, part = tl.program_id(0), tl.program_id(1)
    columns = part * 512 + tl.arange(0, 512)
    logits = tl.load(L + row * STRIDE + columns, columns < WIDTH, other=-float("inf")).to(
        tl.float32
    )
    maximum = tl.max(logits, 0)
    denominator = tl.sum(tl.exp(logits - maximum), 0)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    keys = (ordered & 0xFFFF0000) | (65535 - columns.to(tl.uint32))
    selected = tl.topk(keys, 32)
    tl.store(KEYS + (row * PARTS + part) * 32 + tl.arange(0, 32), selected)
    tl.store(MAXIMUM + row * PARTS + part, maximum)
    tl.store(SUM + row * PARTS + part, denominator)


@triton.jit
def sample_merge_kernel(
    L,
    KEYS,
    MAXIMUM,
    SUM,
    STEP,
    OUT,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_P: tl.constexpr,
    STREAM: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    keys = tl.load(KEYS + row * PARTS * 32 + col, col < PARTS * 32, other=0)
    selected = tl.topk(keys, 32)
    ids = (65535 - (selected & 65535)).to(tl.int32)
    values = tl.load(L + row * STRIDE + ids).to(tl.float32)
    part = tl.arange(0, 16)
    maxima = tl.load(MAXIMUM + row * PARTS + part, part < PARTS, other=-float("inf"))
    sums = tl.load(SUM + row * PARTS + part, part < PARTS, other=0.0)
    maximum = tl.max(maxima, 0)
    denominator = tl.sum(sums * tl.exp(maxima - maximum), 0)
    probability = tl.exp(values - maximum) / denominator
    rank = tl.arange(0, 32)
    kept = ((tl.cumsum(probability, 0) <= TOP_P) & (rank < 30)) | (rank == 0)
    step = tl.load(STEP)
    random = tl.rand(42, (step * 64 * 16 + row * 16 + STREAM) * 32 + rank)
    noise = -tl.log(tl.maximum(random, 1e-7))
    winner = tl.argmax(tl.where(kept, probability, 0.0) / noise, 0)
    choice = tl.sum(tl.where(rank == winner, ids, 0), 0)
    tl.store(OUT + row, choice)


def sample_partitioned(logits, step, stream, top_p=0.9):
    parts = triton.cdiv(logits.shape[1], 512)
    keys = torch.empty((len(logits), parts * 32), dtype=torch.uint32, device=logits.device)
    maxima = torch.empty((len(logits), parts), dtype=torch.float32, device=logits.device)
    sums = torch.empty_like(maxima)
    output = torch.empty(len(logits), dtype=torch.int64, device=logits.device)
    sample_partition_kernel[(len(logits), parts)](
        logits, keys, maxima, sums, logits.stride(0), logits.shape[1], parts, num_warps=4
    )
    sample_merge_kernel[(len(logits),)](
        logits,
        keys,
        maxima,
        sums,
        step,
        output,
        logits.stride(0),
        parts,
        triton.next_power_of_2(parts * 32),
        top_p,
        stream,
        num_warps=4,
    )
    return output


@triton.jit
def attention_metadata_kernel(
    SLOTS, POSITIONS, INDICES, LENGTHS, PAGES: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    pages = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    tl.store(INDICES + row * PAGES + pages, slot * PAGES + pages, pages < PAGES)
    tl.store(LENGTHS + row, tl.load(POSITIONS + row) + 1)


def attention_metadata(slots, positions, indices, lengths, pages):
    attention_metadata_kernel[(len(slots),)](
        slots, positions, indices, lengths, pages, triton.next_power_of_2(pages), num_warps=4
    )


@triton.jit
def expert_counts_kernel(IDS, COUNTS, PAIRS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    pair = tl.arange(0, BLOCK)
    ids = tl.load(IDS + pair, pair < PAIRS, other=-1)
    tl.store(COUNTS + expert, tl.sum((ids == expert).to(tl.int32), 0))


@triton.jit
def expert_offsets_kernel(COUNTS, STARTS, TOTAL):
    expert = tl.arange(0, 256)
    count = tl.load(COUNTS + expert)
    padded = ((count + 15) // 16) * 16
    ends = tl.cumsum(padded, 0)
    tl.store(STARTS + expert, ends - padded)
    tl.store(TOTAL, tl.sum(padded, 0))


@triton.jit
def expert_scatter_kernel(
    IDS,
    COUNTS,
    STARTS,
    SORTED,
    EXPERTS,
    PAIRS: tl.constexpr,
    BLOCK: tl.constexpr,
    EXPERT_BLOCKS: tl.constexpr,
):
    expert = tl.program_id(0)
    pair = tl.arange(0, BLOCK)
    start, count = tl.load(STARTS + expert), tl.load(COUNTS + expert)
    ids = tl.load(IDS + pair, pair < PAIRS, other=-1)
    match = ids == expert
    rank = tl.cumsum(match.to(tl.int32), 0) - 1
    tl.store(SORTED + start + rank, pair, match)
    padded = ((count + 15) // 16) * 16
    padding = tl.arange(0, 16) + count
    tl.store(SORTED + start + padding, PAIRS, padding < padded)
    blocks = tl.arange(0, EXPERT_BLOCKS)
    tl.store(EXPERTS + start // 16 + blocks, expert, blocks * 16 < padded)


def native_moe(x, topk_ids, topk_weights, experts):
    """Fish's FP8 GEMMs with device-only routing layout and 16-row decode tiles."""
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
        fused_moe_gemm,
        quant_kernel,
        quantize_groups,
        silu_mul_quant_kernel,
    )

    tokens = len(x)
    pairs = tokens * 8
    max_rows = triton.cdiv(pairs + min(pairs, 256) * 15, 16) * 16
    counts = torch.empty(256, dtype=torch.int32, device=x.device)
    starts = torch.empty_like(counts)
    total = torch.empty(1, dtype=torch.int32, device=x.device)
    sorted_ids = torch.empty(max_rows, dtype=torch.int32, device=x.device)
    expert_ids = torch.empty(max_rows // 16, dtype=torch.int32, device=x.device)
    expert_counts_kernel[(256,)](topk_ids, counts, pairs, triton.next_power_of_2(pairs))
    expert_offsets_kernel[(1,)](counts, starts, total)
    expert_scatter_kernel[(256,)](
        topk_ids,
        counts,
        starts,
        sorted_ids,
        expert_ids,
        pairs,
        triton.next_power_of_2(pairs),
        triton.next_power_of_2(triton.cdiv(tokens, 16)),
    )
    quantized, scales = quantize_groups(quant_kernel, x, 2048)
    gate_up = torch.empty((pairs, 1024), dtype=x.dtype, device=x.device)
    fused_moe_gemm(
        quantized,
        scales,
        experts.w13_weight,
        experts.w13_scale_inv,
        gate_up,
        None,
        sorted_ids,
        expert_ids,
        total,
        top_k=8,
        block_m=16,
        group_m=1,
    )
    activated, activation_scales = quantize_groups(silu_mul_quant_kernel, gate_up, 512)
    down = torch.empty((pairs, 2048), dtype=torch.float32, device=x.device)
    fused_moe_gemm(
        activated,
        activation_scales,
        experts.w2_weight,
        experts.w2_scale_inv,
        down,
        topk_weights.flatten(),
        sorted_ids,
        expert_ids,
        total,
        top_k=1,
        block_m=16,
        group_m=1,
    )
    return down.view(tokens, 8, 2048).sum(1).to(x.dtype)
