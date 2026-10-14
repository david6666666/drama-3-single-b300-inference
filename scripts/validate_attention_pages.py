"""Check paged attention with changing graph replay lengths and request slots."""

import torch
import torch.nn.functional as F

from drama3.model import AttentionBatch

with torch.inference_mode():
    torch.manual_seed(42)
    keys = torch.randn(8, 512, 2, 256, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    for size in (1, 4):
        slots = torch.arange(size, device="cuda")
        positions = torch.full((size,), 511, device="cuda")
        q = torch.randn(size, 16, 256, device="cuda", dtype=torch.bfloat16)
        print("plan", size, flush=True)
        batch = AttentionBatch(slots, positions, 512)
        print("run", size, flush=True)
        for _ in range(3):
            out = batch.run(q, keys, values)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = batch.run(q, keys, values)
        for length in (1, 16, 17, 33, 301, 512):
            slot_ids = list(range(7, 7 - size, -1))
            batch.plan(slot_ids, [length - 1] * size)
            graph.replay()
            expected = F.scaled_dot_product_attention(
                q[:, :, None, :],
                keys[slot_ids, :length].transpose(1, 2),
                values[slot_ids, :length].transpose(1, 2),
                enable_gqa=True,
            ).squeeze(2)
            torch.testing.assert_close(out, expected, atol=0.015625, rtol=0.02)
            print("PASS", size, length, flush=True)
