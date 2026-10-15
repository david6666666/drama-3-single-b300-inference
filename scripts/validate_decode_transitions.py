"""CPU lifecycle checks for the real Engine.complete_decode transition.

These checks cover host accounting and callback ordering only. They do not
validate CUDA request state, acoustic contents, cancellation or mixed kernels.
"""

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


def request(name, slot, count, position, limit, boundary):
    return SimpleNamespace(
        name=name,
        slot=slot,
        frame_count=count,
        position=position,
        max_frames=limit,
        next_emit=boundary,
    )


def frames(main_tokens):
    values = torch.tensor(main_tokens, dtype=torch.int64)
    output = torch.zeros((*values.shape, 11), dtype=torch.int64)
    output[..., 0] = values
    return output


def validate():
    if torch.cuda.is_initialized():
        raise RuntimeError("This diagnostic must run without an initialized CUDA context")
    # Engine dependencies eagerly query CUDA during import. Hide devices before
    # importing the real class, without replacing or extracting its implementation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from drama3.engine import Engine

    eos = 999
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(config=SimpleNamespace(eos_token_id=eos))
    engine.async_codec = False
    engine.measure = lambda *_: nullcontext()
    emissions, completions, order = [], [], []

    def emit_batch(requests):
        names = [item.name for item in requests]
        emissions.append(names)
        order.append(("emit", names))

    def finish(item):
        completions.append(item.name)
        order.append(("finish", item.name))

    engine.emit_audio_batch = emit_batch
    engine.finish_audio = finish
    a = request("eos_inside_multistep", 3, 4, 10, 50, 6)
    b = request("frame_limit_inside_multistep", 7, 4, 20, 6, 8)
    c = request("audio_boundary_survivor", 2, 3, 30, 20, 6)
    d = request("below_boundary_survivor", 9, 1, 40, 20, 6)
    free = [4]
    active = [a, b, c, d]
    output = frames([[100, 100, 100, 100], [eos, 101, 101, 101], [102, 102, 102, 102]])
    survivors = engine.complete_decode(active, output, free)
    assert survivors == [c, d], "Survivors must retain their original relative order"
    assert free == [4, 3, 7], "EOS and frame-limit slots must each be released exactly once"
    assert [(r.frame_count, r.position) for r in active] == [(5, 11), (6, 22), (6, 33), (4, 43)]
    assert emissions == [[a.name, b.name, c.name]], "Flush final audio even below its boundary"
    assert completions == [a.name, b.name]
    assert order == [("emit", [a.name, b.name, c.name]), ("finish", a.name), ("finish", b.name)]
    first_case = {
        "name": "multistep_eos_limit_boundary_and_survivor_order",
        "passed": True,
        "released_slots": free[1:].copy(),
        "counts_and_positions": [(r.frame_count, r.position) for r in active],
    }

    # The codec callback normally advances this boundary. This check supplies
    # the next boundary explicitly; the stub does not claim codec-state coverage.
    c.next_emit = 12
    order.clear()
    survivors = engine.complete_decode(survivors, frames([103, eos]), free)
    assert survivors == [c]
    assert (c.frame_count, c.position) == (7, 34)
    assert (d.frame_count, d.position) == (4, 43), "An immediate EOS is not an acoustic frame"
    assert free == [4, 3, 7, 9], "Previously finished requests must not re-enter the transition"
    assert emissions[-1] == [d.name]
    assert completions == [a.name, b.name, d.name]
    assert order == [("emit", [d.name]), ("finish", d.name)]
    second_case = {"name": "two_dimensional_single_step_and_immediate_eos", "passed": True}

    # A one-frame remainder ends at the limit without waiting for EOS, while
    # trailing graph outputs cannot increase host counters after completion.
    last = request("one_frame_remainder", 11, 1, 50, 2, 6)
    order.clear()
    survivors = engine.complete_decode([last], frames([[100], [101], [eos], [102]]), free)
    assert survivors == []
    assert (last.frame_count, last.position) == (2, 51)
    assert free == [4, 3, 7, 9, 11]
    assert free.count(11) == 1
    assert order == [("emit", [last.name]), ("finish", last.name)]
    assert not torch.cuda.is_initialized(), "CPU lifecycle checks must not initialize CUDA"
    return [first_case, second_case, {"name": "one_frame_remainder_stops_once", "passed": True}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = Path(__file__).resolve()
    result = {
        "status": "PASS",
        "repository_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path.parents[1], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(path.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (path, path.parents[1] / "src/drama3/engine.py")
        },
        "torch_version": torch.__version__,
        "cuda_initialized": False,
        "cases": validate(),
        "scope": "Host request lifecycle only; no GPU, cancellation or audio-quality claim",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
