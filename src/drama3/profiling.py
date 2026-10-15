"""Opt-in stage diagnostics; never enabled for performance claims."""

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


class TimelineProfiler:
    """Record stream events without synchronizing each stage or request.

    All measure() calls run on the GPU owner thread. Events are resolved once
    at shutdown, after queued PCM delivery has drained. Nested main-stream
    submission and codec-stream work intentionally overlap in the output.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.origin = None
        self.events = []

    @contextmanager
    def measure(self, stage, batch, **metadata):
        if self.origin is None:
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        stream = torch.cuda.current_stream().cuda_stream
        before = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append(
                (
                    stage,
                    batch,
                    stream,
                    before,
                    (time.perf_counter() - before) * 1000,
                    start,
                    end,
                    metadata,
                )
            )

    def save(self):
        torch.cuda.synchronize()
        rows = [
            dict(
                stage=stage,
                batch=batch,
                cuda_stream=stream,
                cpu_start_s=before,
                cpu_submit_ms=cpu_ms,
                gpu_start_ms=self.origin.elapsed_time(start),
                gpu_end_ms=self.origin.elapsed_time(end),
                gpu_ms=start.elapsed_time(end),
                **metadata,
            )
            for stage, batch, stream, before, cpu_ms, start, end, metadata in self.events
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(dict(nonblocking_events_diagnostics_only=True, records=rows), indent=2)
            + "\n"
        )
