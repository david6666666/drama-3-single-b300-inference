"""CPU ownership, capacity and native arithmetic contracts for mixed row reuse."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
reuse = pytest.importorskip("drama3.row_reuse")
MIXED_ROW_REUSE_BUCKETS = reuse.MIXED_ROW_REUSE_BUCKETS
plan_row_reuse = reuse.plan_row_reuse


def prompt(length, value=0):
    result = torch.zeros((length, 11), dtype=torch.int64)
    result[:, 0] = value
    return result


def check_owners(plan, prompts):
    d, full, k = plan.decode_rows, plan.decode_rows + plan.prefill_capacity, plan.compact_rows
    keep = plan.packed_indices[:k].numpy()
    inverse = plan.packed_indices[k:].numpy()
    owner = keep[inverse]
    assert len(inverse) == full and np.all(inverse >= 0) and np.all(inverse < k)
    assert np.array_equal(owner[:d], np.arange(d))
    offsets = [d]
    for p in prompts:
        offsets.append(offsets[-1] + len(p))
    assert np.array_equal(owner[offsets[-1] :], np.arange(offsets[-1], full))
    for sequence, p in enumerate(prompts):
        for position in range(len(p)):
            row = offsets[sequence] + position
            source = int(owner[row])
            assert d <= source <= row
            previous = next(i for i in range(len(prompts)) if offsets[i] <= source < offsets[i + 1])
            assert source - offsets[previous] == position
            assert torch.equal(p[: position + 1], prompts[previous][: position + 1])
    assert len(np.unique(owner)) == plan.unique_rows
    assert np.array_equal(keep[: plan.unique_rows], np.flatnonzero(owner == np.arange(full)))
    assert np.all(keep[plan.unique_rows :] == 0)


@pytest.mark.parametrize("d,capacity,span,compact", MIXED_ROW_REUSE_BUCKETS)
def test_capture_dummy_and_runtime_shape(d, capacity, span, compact):
    sizes = [capacity // 9] * 8
    prompts = [prompt(n, 3) for n in sizes]
    plan = plan_row_reuse(prompts, d, capacity, (compact,))
    assert plan is not None and plan.compact_rows == compact
    assert max([*sizes, capacity - sum(sizes)]) <= span
    check_owners(plan, prompts)


@pytest.mark.parametrize("column", range(11))
def test_every_column_and_causal_history_matters(column):
    a, b = prompt(128), prompt(128)
    b[17, column] = 7
    prompts = [a, b, a.clone(), b.clone()]
    plan = plan_row_reuse(prompts, 64, 768, (576,))
    assert plan is not None
    check_owners(plan, prompts)
    inverse = plan.packed_indices[plan.compact_rows :]
    assert torch.equal(inverse[64:81], inverse[192:209])
    # Later identical rows cannot be merged after the first changed conditioning row.
    assert torch.all(inverse[81:192] != inverse[209:320])


def test_partial_lengths_reordered_admission_and_cancelled_request():
    a, b = prompt(128), prompt(128)
    b[12, 10] = 1
    prompts = [a, b, a[:80], b[:100], a.clone()]
    plan = plan_row_reuse(prompts, 64, 768, (576,))
    assert plan is not None
    check_owners(plan, prompts)
    reversed_prompts = list(reversed(prompts))
    reverse = plan_row_reuse(reversed_prompts, 64, 768, (576,))
    assert reverse is not None
    check_owners(reverse, reversed_prompts)
    assert not torch.equal(plan.packed_indices, reverse.packed_indices)
    # A cancellation changes the roster; rebuild metadata, never reuse its old aliases.
    survivors = [a, a.clone(), a.clone()]
    remaining = plan_row_reuse(survivors, 64, 768, (576,))
    assert remaining is not None
    check_owners(remaining, survivors)
    assert plan_row_reuse(survivors[:2], 64, 768, (576,)) is None


def test_no_alias_and_capacity_fallback():
    distinct = [prompt(128, i) for i in range(6)]
    assert plan_row_reuse(distinct, 64, 768, (576,)) is None
    assert plan_row_reuse(distinct, 64, 768, ()) is None
    # Even identical short prompts retain all original padding; no padding compaction.
    assert plan_row_reuse([prompt(3)] * 8, 64, 768, (576,)) is None


def test_smallest_prebuilt_bucket_covers_ceil64_requirement():
    prompts = [prompt(180), prompt(180), prompt(170, 1), prompt(170, 1)]
    plan = plan_row_reuse(prompts, 64, 1536, (1344, 1088))
    assert plan is not None and plan.compact_rows == 1344
    assert (plan.unique_rows + 63) // 64 * 64 <= plan.compact_rows
    check_owners(plan, prompts)


@pytest.mark.parametrize("bad", [torch.zeros(128, 11), torch.zeros(128, 10, dtype=torch.int64)])
def test_invalid_prompt_contract(bad):
    with pytest.raises(ValueError):
        plan_row_reuse([bad], 64, 768, (576,))


@pytest.mark.parametrize("compact", [0, 575, 832])
def test_invalid_capacity_contract(compact):
    with pytest.raises(ValueError):
        plan_row_reuse([prompt(128)] * 4, 64, 768, (compact,))


def test_indirect_kernel_preserves_native_arithmetic():
    path = Path(__file__).parents[1] / "src/drama3/kernels.py"
    nodes = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    original = nodes["combine_experts_add_norm_kernel"]
    indirect = copy.deepcopy(nodes["combine_experts_add_norm_inverse_kernel"])
    indirect.name = original.name
    indirect.args.args = [a for a in indirect.args.args if a.arg != "INVERSE"]
    indirect.body = [
        n
        for n in indirect.body
        if not (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "source_row" for t in n.targets)
        )
    ]

    class RestoreAddress(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "source_row":
                node.id = "row"
            return node

    RestoreAddress().visit(indirect)
    assert ast.dump(original, include_attributes=False) == ast.dump(
        indirect, include_attributes=False
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("external_numpy_write", [False, True])
def test_stale_plan_rejects_same_length_conditioning_change(external_numpy_write):
    prompts = [prompt(128) for _ in range(4)]
    plan = plan_row_reuse(prompts, 64, 768, (576,))
    assert plan.matches_prompts(prompts)
    if external_numpy_write:
        prompts[1].numpy()[17, 10] = 9
    else:
        prompts[1][17, 10] = 9
    assert not plan.matches_prompts(prompts)


def test_inference_tensor_plan_is_bound_by_content():
    with torch.inference_mode():
        prompts = [prompt(128) for _ in range(4)]
        plan = plan_row_reuse(prompts, 64, 768, (576,))
        assert plan.matches_prompts(prompts)
        changed = [p.clone() for p in prompts]
        changed[0][31, 1] = 3
        assert not plan.matches_prompts(changed)


@pytest.fixture
def graph_harness():
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / "src/drama3/mixed_graph.py"
    tree = ast.parse(path.read_text())
    graph = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MixedGraph")
    prepare = next(n for n in graph.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    namespace = {"torch": torch, "plan_row_reuse": plan_row_reuse}
    exec(compile(ast.Module(body=[prepare], type_ignores=[]), str(path), "exec"), namespace)
    harness = SimpleNamespace(
        model=SimpleNamespace(slots=256),
        decode_size=64,
        batch=8,
        capacity=768,
        compact_rows=576,
        row_indices=torch.full((576 + 832,), -1, dtype=torch.int64),
        slots=torch.zeros(832, dtype=torch.int64),
        frames=torch.zeros((832, 11), dtype=torch.int64),
        layout=SimpleNamespace(
            max_length=256,
            boundaries=torch.zeros(10, dtype=torch.int32),
            sequence_slots=torch.zeros(9, dtype=torch.int64),
            update_chunks=lambda lengths: None,
        ),
    )
    harness.prepare = lambda active, prompts, slots, plan=None: namespace["prepare"](
        harness, active, prompts, slots, plan
    )
    active = [SimpleNamespace(slot=i) for i in range(56)]
    return harness, active


def test_actual_prepare_rejects_stale_plan_before_writing_graph(graph_harness):
    graph, active = graph_harness
    prompts = [prompt(128) for _ in range(4)]
    plan = plan_row_reuse(prompts, 64, 768, (576,))
    graph.prepare(active, prompts, [56, 57, 58, 59], plan)
    before = [x.clone() for x in (graph.row_indices, graph.slots, graph.frames)]
    prompts[1][17, 10] = 9
    with pytest.raises(ValueError, match="Prompts do not match"):
        graph.prepare(active, prompts, [56, 57, 58, 59], plan)
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (graph.row_indices, graph.slots, graph.frames), strict=True)
    )


def test_actual_prepare_checks_slot_ownership_and_original_fallback(graph_harness):
    graph, active = graph_harness
    prompts = [prompt(128) for _ in range(4)]
    with pytest.raises(ValueError, match="disjoint"):
        graph.prepare(active, prompts, [0, 57, 58, 59])
    graph.compact_rows = None
    distinct = [prompt(128, i) for i in range(4)]
    assert plan_row_reuse(distinct, 64, 768, (576,)) is None
    graph.prepare(active, distinct, [56, 57, 58, 59])
    assert torch.equal(graph.frames[64:192], distinct[0])
