"""Validate mixed scheduling with real inputs and identical-input layer controls.

This diagnostic fixes RRMS < 0.03 before execution. Layer comparisons, graph
replay comparisons, and exact state ownership checks are separate gates; none
substitutes for the full-corpus WER/SIM or service throughput acceptance tests.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

import drama3.mixed_model as mixed_model
from drama3.engine import BatchedPrefillGraph, DecodeGraph, state_pool_size
from drama3.mixed_graph import MixedGraph
from drama3.model import DramaModel
from drama3.runtime_state import DeviceRequests


class Checks:
    def __init__(self, output, metadata):
        self.output = output
        self.report = dict(metadata=metadata, cases=[], checks=[], status="RUNNING")
        self.flush()

    def flush(self):
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def tensor(self, name, actual, expected, *, exact=False, per_slot=False):
        if actual.shape != expected.shape:
            result = dict(
                name=name, passed=False, shape=list(actual.shape), expected=list(expected.shape)
            )
        elif exact:
            result = dict(name=name, exact=True, passed=torch.equal(actual, expected))
        else:
            finite = bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
            delta = (actual.float() - expected.float()).square()
            reference = expected.float().square()
            error = float((delta.mean() / reference.mean().clamp_min(1e-12)).sqrt())
            worst = error
            if per_slot and len(actual):
                rows = delta.reshape(len(actual), -1).mean(-1)
                norms = reference.reshape(len(actual), -1).mean(-1).clamp_min(1e-12)
                worst = float((rows / norms).sqrt().max())
            result = dict(
                name=name,
                exact=False,
                rrms=error if math.isfinite(error) else None,
                worst_slot_rrms=worst if math.isfinite(worst) else None,
                passed=finite and error < 0.03 and worst < 0.03,
            )
        self.report["checks"].append(result)
        self.flush()
        if not result["passed"]:
            print(json.dumps(result), flush=True)
            raise AssertionError(result)


def snapshot(tensors):
    return [value.clone() for value in tensors]


def restore(tensors, saved):
    for value, original in zip(tensors, saved, strict=True):
        value.copy_(original)


def state_tensors(model, device):
    result = {}
    for index, pair in model.states.items():
        result[f"conv/{index}"], result[f"gdn/{index}"] = pair
    for index, pair in model.kv.items():
        result[f"key/{index}"], result[f"value/{index}"] = pair
    result.update(
        request_last=device.last,
        request_history=device.history,
        request_metadata=device.metadata,
        request_codes=device.codes,
        sampling_step=model.sampling_step,
    )
    return result


def inspect_layers(model, graph, checks, prefix, owned, untouched):
    """Run one eager trajectory, restoring each candidate state after its probe."""
    original_linear = mixed_model.mixed_linear_attention
    original_attention = mixed_model.mixed_attention
    original_moe = model.moe
    layer_indices = {id(layer.mlp): i for i, layer in enumerate(model.backbone.layers)}
    visited = dict(gdn=[], attention=[], moe=[])

    def ownership(name, tensors, before):
        for kind, value, saved in zip(("first", "second"), tensors, before, strict=True):
            checks.tensor(
                f"{prefix}/{name}/untouched_{kind}",
                value.index_select(0, untouched),
                saved.index_select(0, untouched),
                exact=True,
            )

    def outputs(name, actual, expected, d):
        checks.tensor(f"{prefix}/{name}/decode", actual[:d], expected[:d], per_slot=True)
        checks.tensor(f"{prefix}/{name}/prefill", actual[d:], expected[d:], per_slot=True)

    def linear_probe(m, x, layer, index, slots, layout, d):
        saved_x = x.clone()
        tensors = m.states[index]
        before = snapshot(tensors)
        actual = original_linear(m, x, layer, index, slots, layout, d)
        candidate = snapshot(tensors)
        try:
            ownership(f"gdn/{index}/candidate", tensors, before)
            restore(tensors, before)
            expected = torch.cat(
                (
                    m.linear_attention(saved_x[:d], layer, index, slots[:d], True),
                    m.linear_attention(saved_x[d:], layer, index, slots[d:], False, layout),
                )
            )
            outputs(f"gdn/{index}/output", actual, expected, d)
            for kind, value, reference in zip(("conv", "state"), candidate, tensors, strict=True):
                checks.tensor(
                    f"{prefix}/gdn/{index}/{kind}",
                    value.index_select(0, owned),
                    reference.index_select(0, owned),
                    per_slot=True,
                )
            ownership(f"gdn/{index}/split", tensors, before)
            visited["gdn"].append(index)
        finally:
            restore(tensors, candidate)
        return actual

    def attention_probe(m, x, layer, index, positions, slots, batch, layout, d):
        saved_x = x.clone()
        tensors = m.kv[index]
        before = snapshot(tensors)
        actual = original_attention(m, x, layer, index, positions, slots, batch, layout, d)
        candidate = snapshot(tensors)
        try:
            ownership(f"attention/{index}/candidate", tensors, before)
            restore(tensors, before)
            expected = torch.cat(
                (
                    m.attention(saved_x[:d], layer, index, positions[:d], slots[:d], batch),
                    m.attention(saved_x[d:], layer, index, positions[d:], slots[d:], None, layout),
                )
            )
            outputs(f"attention/{index}/output", actual, expected, d)
            for kind, value, reference in zip(("key", "value"), candidate, tensors, strict=True):
                checks.tensor(
                    f"{prefix}/attention/{index}/{kind}_written_rows",
                    value[slots, positions],
                    reference[slots, positions],
                    per_slot=True,
                )
            ownership(f"attention/{index}/split", tensors, before)
            visited["attention"].append(index)
        finally:
            restore(tensors, candidate)
        return actual

    def moe_probe(x, layer, decode=False):
        if not decode:
            raise AssertionError("Mixed backbone must retain the full native decode MoE path")
        d = graph.decode_size
        saved_x = x.clone()
        actual = original_moe(x, layer, decode=True)
        expected = torch.cat(
            (
                original_moe(saved_x[:d], layer, decode=True),
                original_moe(saved_x[d:], layer, decode=True),
            )
        )
        index = layer_indices[id(layer)]
        outputs(f"moe/{index}", actual, expected, d)
        visited["moe"].append(index)
        return actual

    mixed_model.mixed_linear_attention = linear_probe
    mixed_model.mixed_attention = attention_probe
    model.moe = moe_probe
    try:
        output = graph.forward()
    finally:
        mixed_model.mixed_linear_attention = original_linear
        mixed_model.mixed_attention = original_attention
        model.moe = original_moe
    expected_indices = list(range(len(model.backbone.layers)))
    if sorted(visited["gdn"] + visited["attention"]) != expected_indices:
        raise AssertionError(f"Incomplete backbone layer coverage: {visited}")
    if visited["moe"] != expected_indices:
        raise AssertionError(f"Incomplete MoE layer coverage: {visited}")
    return output, visited


def parse_cases(values):
    cases = []
    for value in values:
        active, arriving = map(int, value.split(":"))
        if not (9 <= active <= 64 and 1 <= arriving <= 8 and active + arriving <= 64):
            raise ValueError("Cases require 9..64 active, 1..8 arriving and at most 64 total")
        cases.append((active, arriving))
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["58:6"], help="Active:arriving counts")
    args = parser.parse_args()
    try:
        cases = parse_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    data = args.corpus.read_bytes()
    samples = [json.loads(row) for row in data.splitlines()][:64]
    if len(samples) != 64:
        parser.error("The diagnostic requires at least 64 real corpus requests")
    repo = Path(__file__).resolve().parents[1]
    properties = torch.cuda.get_device_properties(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = Checks(
        args.output,
        dict(
            repository_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True
            ),
            corpus_sha256=hashlib.sha256(data).hexdigest(),
            source_sha256={
                name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/validate_mixed_layers.py",
                    "src/drama3/mixed_model.py",
                    "src/drama3/mixed_graph.py",
                )
            },
            checkpoint=args.checkpoint,
            cases=args.cases,
            device_name=properties.name,
            device_uuid=str(getattr(properties, "uuid", "unavailable")),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__,
            environment={k: v for k, v in sorted(os.environ.items()) if k.startswith("DRAMA3_")},
            rrms_limit=0.03,
            per_row_or_slot_rrms_limit=0.03,
            ownership="exact equality for every untouched state slot",
            scope="layer parity and graph parity only; no serving or WER/SIM acceptance claim",
        ),
    )
    try:
        print(json.dumps(dict(stage="capture", cases=args.cases)), flush=True)
        model = DramaModel(args.checkpoint, slots=state_pool_size(64), max_length=2048)
        if model.moe_shared_stream is not None or model.tuned_prefill_moe:
            raise ValueError("Use shared overlap=0 and tuned prefill MoE=0 for this control")
        if model.prefill_moe != "flashinfer":
            raise ValueError("Set DRAMA3_PREFILL_MOE=flashinfer for both native MoE arms")
        device = DeviceRequests(model.slots, model.config.eos_token_id, model.max_length)
        prompts = [
            build_drama3_prompt(
                sample["text"],
                model.weights.tokenizer,
                model.config,
                [sample["reference_text"]],
                [torch.tensor(sample["reference_codes"])],
            )
            for sample in samples
        ]
        prefill = BatchedPrefillGraph(model, 8, triton.next_power_of_2(max(map(len, prompts))))
        first = DecodeGraph(model, 8, first_frame=True, requests=device)
        graphs = []
        for active_count, arriving_count in cases:
            selected = prompts[active_count : active_count + arriving_count]
            capacity = triton.cdiv(sum(map(len, selected)), 256) * 256
            graphs.append(
                MixedGraph(model, device, triton.next_power_of_2(active_count), 8, capacity)
            )
        # Every capture above may update state. Initialize real requests only
        # afterwards; later cases deliberately reuse physical slots from earlier ones.
        tensors = state_tensors(model, device)
        for case_index, ((active_count, arriving_count), graph) in enumerate(
            zip(cases, graphs, strict=True)
        ):
            prefix = f"case{case_index}/d{active_count}-p{arriving_count}"
            print(json.dumps(dict(stage="initialize", case=prefix)), flush=True)
            active = [
                SimpleNamespace(slot=i, position=len(prompts[i]) - 1) for i in range(active_count)
            ]
            for index, prompt in enumerate(prompts[:active_count]):
                history = F.pad(prompt[-16:, 0], (max(0, 16 - len(prompt)), 0), value=-1).cuda()
                device.initialize(index, prompt[-1].cuda(), history, len(prompt) - 1, 500)
            for start in range(0, active_count, 8):
                end = min(start + 8, active_count)
                hidden = prefill.run(prompts[start:end], list(range(start, end)))
                first.run(active[start:end], hidden)
            arriving_slots = list(range(active_count, active_count + arriving_count))
            graph.prepare(
                active, prompts[active_count : active_count + arriving_count], arriving_slots
            )
            decode_slots = graph.slots[: graph.decode_size].cpu().tolist()
            prefill_slots = graph.layout.sequence_slots.cpu().tolist()
            owned_ids = sorted(set(decode_slots + prefill_slots))
            if len(owned_ids) != len(decode_slots) + len(prefill_slots):
                raise AssertionError("Decode and prefill state slots overlap")
            owned = torch.tensor(owned_ids, device="cuda", dtype=torch.int64)
            untouched = torch.tensor(
                sorted(set(range(model.slots)) - set(owned_ids)), device="cuda", dtype=torch.int64
            )
            row = dict(
                case=prefix,
                decode_rows=graph.decode_size,
                active_count=active_count,
                arriving_count=arriving_count,
                capacity=graph.capacity,
                decode_slots=decode_slots,
                prefill_sequence_slots=prefill_slots,
                slot_reuse=case_index > 0,
                passed=False,
            )
            checks.report["cases"].append(row)
            checks.flush()
            initial = snapshot(tensors.values())
            graph.graph.replay()
            captured_output = tuple(value.clone() for value in graph.output)
            captured_state = snapshot(tensors.values())
            for (name, value), before in zip(tensors.items(), initial, strict=True):
                if name.startswith(("conv/", "gdn/", "key/", "value/")):
                    checks.tensor(
                        f"{prefix}/graph/untouched/{name}",
                        value.index_select(0, untouched),
                        before.index_select(0, untouched),
                        exact=True,
                    )
                else:
                    checks.tensor(f"{prefix}/graph/read_only/{name}", value, before, exact=True)
            restore(tensors.values(), initial)
            print(json.dumps(dict(stage="layer_checks", case=prefix)), flush=True)
            eager_output, coverage = inspect_layers(model, graph, checks, prefix, owned, untouched)
            row["layer_coverage"] = coverage
            for name, actual, expected in zip(
                ("decode", "prefill"), captured_output, eager_output, strict=True
            ):
                checks.tensor(f"{prefix}/graph_vs_eager/{name}", actual, expected, per_slot=True)
            for (name, value), captured, before in zip(
                tensors.items(), captured_state, initial, strict=True
            ):
                if name.startswith(("conv/", "gdn/", "key/", "value/")):
                    checks.tensor(
                        f"{prefix}/eager/untouched/{name}",
                        value.index_select(0, untouched),
                        before.index_select(0, untouched),
                        exact=True,
                    )
                    checks.tensor(
                        f"{prefix}/graph_vs_eager/{name}",
                        captured.index_select(0, owned),
                        value.index_select(0, owned),
                        per_slot=True,
                    )
                else:
                    checks.tensor(f"{prefix}/graph_vs_eager/{name}", captured, value, exact=True)
            row["passed"] = True
            checks.flush()
            print(
                json.dumps(
                    dict(stage="case_passed", case=prefix, checks=len(checks.report["checks"]))
                ),
                flush=True,
            )
            del initial, captured_state, captured_output
        checks.report["status"] = "PASS"
        checks.flush()
    except Exception as error:
        checks.report["status"] = "FAIL"
        checks.report["failure"] = dict(type=type(error).__name__, message=str(error))
        checks.flush()
        raise


if __name__ == "__main__":
    main()
