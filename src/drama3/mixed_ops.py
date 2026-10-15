"""Native mixed-row normalization without concatenation or a strided gate copy."""

import torch
import triton
import triton.language as tl


@triton.jit
def mixed_gated_norm_kernel(
    DECODE, PREFILL, Z, W, OUT, DECODE_ROWS: tl.constexpr, Z_STRIDE: tl.constexpr, EPS: tl.constexpr
):
    group = tl.program_id(0)
    d = tl.arange(0, 128)
    if group < DECODE_ROWS * 32:
        x = tl.load(DECODE + group * 128 + d).to(tl.float32)
    else:
        x = tl.load(PREFILL + (group - DECODE_ROWS * 32) * 128 + d).to(tl.float32)
    # Preserve all BF16 rounding boundaries and the original reduction geometry.
    norm = x * tl.rsqrt(tl.sum(x * x, 0) / 128 + EPS)
    norm = norm.to(OUT.dtype.element_ty).to(tl.float32)
    scaled = (norm * tl.load(W + d).to(tl.float32)).to(OUT.dtype.element_ty).to(tl.float32)
    z = tl.load(Z + (group // 32) * Z_STRIDE + (group % 32) * 128 + d).to(tl.float32)
    tl.store(OUT + group * 128 + d, scaled * (z * tl.sigmoid(z)))


def mixed_gated_norm(decode, prefill, z, weight, epsilon):
    """Consume the two existing output buffers and strided projection gates directly."""
    output = torch.empty((len(z), 4096), dtype=decode.dtype, device=decode.device)
    mixed_gated_norm_kernel[(len(z) * 32,)](
        decode,
        prefill,
        z,
        weight,
        output,
        len(decode),
        z.stride(0),
        epsilon,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
