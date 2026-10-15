"""CPU scheduling/completion boundaries; full model parity is a GPU validation gate."""

import ast
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

TORCH = pytest.importorskip("torch")

pipeline = pytest.importorskip("drama3.frame_pipeline")


def request(slot=0, *, count=4, limit=100, emit=6):
    return SimpleNamespace(
        slot=slot, frame_count=count, position=100 + count, max_frames=limit, next_emit=emit
    )


@pytest.mark.parametrize(
    ("waiting", "pending", "supported", "emit", "limit", "expected"),
    [
        (False, False, True, 6, 100, None),
        (False, False, True, 20, 6, None),
        (False, False, True, 5, 100, "audio_boundary"),
        (False, False, True, 20, 5, "audio_boundary"),
        (True, False, True, 6, 100, "admission"),
        (False, True, True, 6, 100, "admission"),
        (False, False, False, 6, 100, "capacity"),
    ],
)
def test_pair_selection(waiting, pending, supported, emit, limit, expected):
    assert (
        pipeline.pair_blocker(
            [request(), request(1, emit=emit, limit=limit)],
            waiting=waiting,
            pending=pending,
            supported=supported,
        )
        == expected
    )


@pytest.fixture
def completion():
    # Importing Engine also loads the installed GPU model stack. Extract the actual
    # CPU-only method so these boundary tests need neither a GPU nor model packages.
    path = Path(__file__).parents[1] / "src/drama3/engine.py"
    tree = ast.parse(path.read_text())
    engine = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    method = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "complete_decode"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)

    class Harness:
        complete_decode = namespace["complete_decode"]

        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(eos_token_id=0))
            self.async_codec = True
            self.emissions = []
            self.finished = []

        def measure(self, *args):
            return nullcontext()

        def emit_audio_batch(self, requests):
            self.emissions.extend((r.slot, r.frame_count) for r in requests)

        def finish_audio(self, item):
            self.finished.append(item.slot)

    return Harness()


@pytest.mark.parametrize(
    "first,second,limit,emit", list(product((0, 7), (0, 11), (5, 6, 100), (5, 6, 100)))
)
def test_pair_completion_stops_at_first_eos_or_limit(completion, first, second, limit, emit):
    item = request(limit=limit, emit=emit)
    frames = TORCH.ones((2, 1, 11), dtype=TORCH.int64)
    frames[:, 0, 0] = TORCH.tensor([first, second])
    free = []
    survivors = completion.complete_decode([item], frames, free)
    accepted = 0 if first == 0 else 1 if second == 0 or limit == 5 else 2
    done = first == 0 or second == 0 or 4 + accepted >= limit
    assert item.frame_count == 4 + accepted
    assert item.position == 104 + accepted
    assert completion.finished == free == ([0] if done else [])
    assert survivors == ([] if done else [item])
    assert completion.emissions == ([(0, 4 + accepted)] if done or 4 + accepted >= emit else [])


def test_pair_completion_preserves_other_rows(completion):
    items = [request(i) for i in range(3)]
    frames = TORCH.ones((2, 3, 11), dtype=TORCH.int64)
    frames[:, :, 0] = TORCH.tensor([[0, 7, 7], [11, 0, 11]])
    free = []
    survivors = completion.complete_decode(items, frames, free)
    assert [r.frame_count for r in items] == [4, 5, 6]
    assert [r.position for r in items] == [104, 105, 106]
    assert free == completion.finished == [0, 1]
    assert survivors == [items[2]]
    assert completion.emissions == [(0, 4), (1, 5), (2, 6)]
