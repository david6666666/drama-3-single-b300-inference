"""Opt-in synchronized stage diagnostics; never enabled for performance claims."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class StageProfiler:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    @contextmanager
    def measure(self, stage, batch):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        before = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            self.rows.append(
                {
                    "stage": stage,
                    "batch": batch,
                    "gpu_timeline_ms": start.elapsed_time(end),
                    "wall_ms": (time.perf_counter() - before) * 1000,
                }
            )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"synchronized_diagnostics_only": True, "records": self.rows}, indent=2)
            + "\n"
        )
