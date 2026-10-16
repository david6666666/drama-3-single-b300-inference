"""Exact causal row ownership for fresh mixed-prefill slots, built only on CPU."""

from dataclasses import dataclass

import numpy as np
import torch

# Five eager captures before readiness. Other shapes use the original mixed graph.
MIXED_ROW_REUSE_BUCKETS = (
    (64, 768, 256, 576),
    (64, 1024, 256, 832),
    (64, 1280, 256, 1088),
    (64, 1536, 256, 1088),
    (64, 1536, 256, 1344),
)


@dataclass(frozen=True)
class RowReusePlan:
    decode_rows: int
    prefill_capacity: int
    compact_rows: int
    unique_rows: int
    prompt_lengths: tuple[int, ...]
    packed_indices: torch.Tensor
    prompt_bytes: tuple[bytes, ...]

    def matches_prompts(self, prompts):
        # Inference tensors have no mutation version; exact bytes also catch writes
        # through external NumPy views. A same-length stale plan must never run.
        return tuple(map(len, prompts)) == self.prompt_lengths and all(
            prompt.device.type == "cpu"
            and prompt.dtype == torch.int64
            and prompt.ndim == 2
            and prompt.shape[1] == 11
            and prompt.numpy().tobytes() == snapshot
            for prompt, snapshot in zip(prompts, self.prompt_bytes, strict=True)
        )

    @property
    def removed_rows(self):
        return self.decode_rows + self.prefill_capacity - self.compact_rows


def plan_row_reuse(prompts, decode_rows, prefill_capacity, compact_capacities):
    """Return a guarded-capacity plan, or None when no prebuilt bucket fits.

    Full 11-column integer rows are compared through the first difference, with
    no hash or digest. Every prompt starts at position zero in a fresh slot;
    decode and original padding rows keep their own owners. Only stateless MoE
    work is shared: attention, recurrent updates and teacher work remain full.
    The plan is consumed immediately with these same unmodified CPU prompts.
    """
    if decode_rows < 1 or prefill_capacity < 1 or not 1 <= len(prompts) <= 8:
        raise ValueError("Invalid mixed row counts")
    if any(
        p.device.type != "cpu" or p.dtype != torch.int64 or p.ndim != 2 or p.shape[1] != 11
        for p in prompts
    ):
        raise ValueError("Row reuse requires CPU int64 prompts with eleven columns")
    lengths = tuple(map(len, prompts))
    if min(lengths) < 3 or max(lengths) > 512 or sum(lengths) > prefill_capacity:
        raise ValueError("Prompts do not fit the mixed prefill capacity")
    capacities = sorted(set(compact_capacities))
    full_rows = decode_rows + prefill_capacity
    if any(k <= 0 or k % 64 or k > full_rows - 256 for k in capacities):
        raise ValueError("Compact buckets must be 64-aligned and remove at least 256 rows")
    if not capacities:
        return None
    arrays = [prompt.numpy() for prompt in prompts]
    rows = np.arange(full_rows, dtype=np.int64)
    owners = rows.copy()
    offsets = []
    offset = decode_rows
    for index, current in enumerate(arrays):
        longest, previous_owner = 0, None
        for previous, other in enumerate(arrays[:index]):
            limit = min(len(current), len(other))
            different = np.flatnonzero(~np.all(current[:limit] == other[:limit], axis=1))
            common = int(different[0]) if len(different) else limit
            if common > longest:
                longest, previous_owner = common, previous
        if longest:
            start = offsets[previous_owner]
            # Earlier owners are already canonical, so this cannot create alias chains.
            owners[offset : offset + longest] = owners[start : start + longest]
        offsets.append(offset)
        offset += len(current)
    keep = np.flatnonzero(owners == rows)
    unique = len(keep)
    minimum = (unique + 63) // 64 * 64
    compact = next((k for k in capacities if k >= minimum), None)
    if compact is None:
        return None
    lookup = np.empty_like(owners)
    lookup[keep] = np.arange(unique, dtype=np.int64)
    packed = np.zeros(compact + full_rows, dtype=np.int64)
    packed[:unique] = keep
    packed[compact:] = lookup[owners]
    # Extra compact rows duplicate row0; no new expert or model state is introduced.
    return RowReusePlan(
        decode_rows,
        prefill_capacity,
        compact,
        unique,
        lengths,
        torch.from_numpy(packed),
        tuple(array.tobytes() for array in arrays),
    )
