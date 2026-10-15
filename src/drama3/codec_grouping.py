"""Cost-gated adjacent codec buckets within one already-ready audio emit."""

import math


def _execution_cost(graphs, frames, count):
    remaining, calls, total = count, 0, 0.0
    for planned in graphs.split(frames, count):
        actual = min(planned, remaining)
        if actual <= 0:
            return None
        padded = next((size for size in graphs.batch_buckets[frames] if size >= actual), None)
        cost = graphs.cost_ms.get((padded, frames, 0))
        if cost is None or not math.isfinite(cost) or cost <= 0:
            return None
        calls += 1
        total += cost
        remaining -= actual
    return (calls, total) if remaining == 0 else None


def coalesce_codec_groups(groups, graphs):
    """Merge disjoint adjacent captured buckets only if fewer calls cost less.

    Jobs stay within the caller's current emit and retain their original tuples.
    Existing split decisions and actual graph batch rounding remain unchanged.
    A merge spans at most one step of the captured frame-bucket list; no new
    shapes, waiting, chunk boundaries, request state or sampling are introduced.
    """
    original = list(groups)
    if len(original) < 2:
        return list(groups.items())
    order = {frame: i for i, frame in enumerate(graphs.frame_buckets)}
    frames = sorted(frame for frame in original if frame in order)
    costs = {frame: _execution_cost(graphs, frame, len(groups[frame])) for frame in frames}
    # Weighted matching on a path: each frame bucket is alone or paired once.
    best = [(0.0, [])]
    for index, frame in enumerate(frames):
        previous, pairs = best[-1]
        chosen = (previous, pairs)
        if index:
            lower = frames[index - 1]
            left, right = costs[lower], costs[frame]
            if order[frame] == order[lower] + 1 and left is not None and right is not None:
                merged = _execution_cost(graphs, frame, len(groups[lower]) + len(groups[frame]))
                if merged is not None and merged[0] < left[0] + right[0]:
                    saving = left[1] + right[1] - merged[1]
                    candidate = best[index - 1][0] + saving
                    if saving > 0 and candidate > chosen[0]:
                        chosen = (candidate, [*best[index - 1][1], (lower, frame)])
        best.append(chosen)
    ownership = {frame: pair for pair in best[-1][1] for frame in pair}
    emitted, result = set(), []
    for frame in original:
        pair = ownership.get(frame)
        if pair is None:
            result.append((frame, groups[frame]))
        elif pair not in emitted:
            # Preserve the original bucket order inside each merged batch.
            jobs = [job for member in original if member in pair for job in groups[member]]
            result.append((pair[1], jobs))
            emitted.add(pair)
    return result
