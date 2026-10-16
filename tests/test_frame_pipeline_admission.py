"""CPU-only checks of the production pair selector and admission boundaries."""

import ast
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("drama3.frame_pipeline")


@pytest.fixture
def selector(monkeypatch):
    # Execute the actual selector and counter initialization without importing the
    # GPU-dependent Engine module. The clock is controlled at the admission edge.
    path = Path(__file__).parents[1] / "src/drama3/engine.py"
    tree = ast.parse(path.read_text())
    engine = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    methods = {node.name: node for node in engine.body if isinstance(node, ast.FunctionDef)}
    attributes = {
        "frame_pipeline",
        "frame_pipeline_deferred_waiting",
        "frame_pipeline_counts",
        "frame_pipeline_waiting_counts",
    }
    initializers = [
        node
        for node in methods["__init__"].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr in attributes
            for target in node.targets
        )
    ]
    namespace = {"os": os, "time": SimpleNamespace(monotonic=lambda: 0.1)}
    module = ast.fix_missing_locations(
        ast.Module(body=[methods["select_frame_pipeline"]], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)

    class Harness:
        select_frame_pipeline = namespace["select_frame_pipeline"]

        def __init__(self, deferred):
            monkeypatch.delenv("DRAMA3_FRAME_PIPELINE", raising=False)
            monkeypatch.delenv("DRAMA3_FRAME_PIPELINE_DEFERRED_WAITING", raising=False)
            if deferred:
                monkeypatch.setenv("DRAMA3_FRAME_PIPELINE_DEFERRED_WAITING", "1")
            scope = {"self": self, "os": os}
            init = ast.fix_missing_locations(ast.Module(body=initializers, type_ignores=[]))
            exec(compile(init, str(path), "exec"), scope)
            self.pending = queue.Queue()
            self.pipeline_graphs = {32: object(), 64: object()}
            self.admission_min_running = 32
            self.admission_min_batch = 6
            self.admission_max_wait = 0.1

    return Harness


def roster(count, *, emit=6, limit=100):
    return [SimpleNamespace(frame_count=4, next_emit=emit, max_frames=limit) for _ in range(count)]


def waiting(count, *, queued_at=0.001):
    return [SimpleNamespace(queued_at=queued_at) for _ in range(count)]


@pytest.mark.parametrize(
    "deferred,active_count,waiting_count,queued_at,free,expected",
    [
        (False, 32, 5, 0.001, [0], "waiting_deferred"),
        (True, 32, 5, 0.001, [0], "eligible"),
        (True, 31, 5, 0.001, [0], "waiting_runnable"),
        (True, 32, 6, 0.001, [0], "waiting_runnable"),
        (True, 32, 1, 0.000001, [0], "eligible"),
        (True, 32, 1, 0.0, [0], "waiting_runnable"),
        (True, 32, 1, -0.001, [0], "waiting_runnable"),
        (True, 31, 6, -0.001, [], "eligible"),
        (False, 31, 6, -0.001, [], "waiting_deferred"),
        (False, 32, 0, 0.0, [0], "eligible"),
        (True, 32, 0, 0.0, [], "eligible"),
    ],
)
def test_runnable_admission_boundaries(
    selector, deferred, active_count, waiting_count, queued_at, free, expected
):
    engine = selector(deferred)
    graph = engine.select_frame_pipeline(
        roster(active_count), waiting(waiting_count, queued_at=queued_at), free, 32
    )
    assert graph is (engine.pipeline_graphs[32] if expected == "eligible" else None)
    assert engine.frame_pipeline_counts[expected] == 1
    assert sum(engine.frame_pipeline_counts.values()) == 1
    assert engine.frame_pipeline is False  # Both new scheduling options remain opt-in.
    assert engine.frame_pipeline_deferred_waiting is deferred


@pytest.mark.parametrize("deferred,waiting_count", [(False, 0), (False, 5), (True, 0), (True, 5)])
def test_pending_always_blocks(selector, deferred, waiting_count):
    engine = selector(deferred)
    engine.pending.put(object())
    assert engine.select_frame_pipeline(roster(32), waiting(waiting_count), [0], 32) is None
    assert engine.frame_pipeline_counts["pending"] == 1
    assert engine.frame_pipeline_waiting_counts["both" if waiting_count else "pending_only"] == 1
    assert engine.pending.qsize() == 1  # Selection does not drain or admit requests.


@pytest.mark.parametrize("emit,limit", [(5, 100), (20, 5), (4, 100)])
def test_deferred_admission_preserves_audio_and_max_frame_horizon(selector, emit, limit):
    engine = selector(True)
    assert (
        engine.select_frame_pipeline(roster(32, emit=emit, limit=limit), waiting(5), [0], 32)
        is None
    )
    assert engine.frame_pipeline_counts["audio_boundary"] == 1
    assert engine.frame_pipeline_waiting_counts["eligible_with_deferred_waiting"] == 0


def test_unsupported_capacity_has_one_decision_and_no_supported_context(selector):
    engine = selector(True)
    engine.pending.put(object())
    assert engine.select_frame_pipeline(roster(8), waiting(6), [0], 8) is None
    assert engine.frame_pipeline_counts["capacity"] == 1
    assert sum(engine.frame_pipeline_counts.values()) == 1
    assert engine.frame_pipeline_waiting_counts["both"] == 0


def test_context_reasons_overlap_without_inflating_decision_count(selector):
    engine = selector(True)
    assert engine.select_frame_pipeline(roster(31), waiting(6, queued_at=0.0), [0], 32) is None
    assert engine.frame_pipeline_counts["waiting_runnable"] == 1
    assert sum(engine.frame_pipeline_counts.values()) == 1
    context = engine.frame_pipeline_waiting_counts
    assert context["waiting_only"] == 1
    assert context["runnable_active_threshold"] == 1
    assert context["runnable_batch_threshold"] == 1
    assert context["runnable_deadline"] == 1


def test_selected_deferred_waiting_context_records_size_and_age(selector):
    engine = selector(True)
    for count, queued_at in [(5, 0.001), (3, 0.05), (5, 0.075)]:
        assert engine.select_frame_pipeline(
            roster(32), waiting(count, queued_at=queued_at), [0], 32
        )
    assert engine.select_frame_pipeline(roster(32), [], [0], 32)
    context = engine.frame_pipeline_waiting_counts
    assert engine.frame_pipeline_counts["eligible"] == 4
    assert context["waiting_only"] == context["eligible_with_deferred_waiting"] == 3
    assert context["neither"] == context["eligible_without_waiting"] == 1
    assert context["deferred_waiting_size"] == {5: 2, 3: 1}
    assert context["deferred_waiting_age_ms_sum"] == pytest.approx(174.0)
    assert context["deferred_waiting_age_ms_max"] == pytest.approx(99.0)
