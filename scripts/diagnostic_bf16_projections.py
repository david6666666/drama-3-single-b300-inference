"""Tune native BF16 projection tiles on real decode inputs, then check every call."""

import json
from functools import partial
from itertools import product

import torch
import triton

from drama3.bf16_projection import projection
from drama3.kernels import linear


def error(actual, expected):
    return float(
        (
            (actual.float() - expected.float()).square().mean()
            / expected.float().square().mean().clamp_min(1e-12)
        ).sqrt()
    )


@torch.inference_mode()
def benchmark(calls, output):
    # Each input is the actual output of the preceding decode operation. Keep
    # a stable copy so candidate launches cannot overwrite captured storage.
    calls = [(x.clone(), weight) for x, weight in calls.values()]
    representatives = {}
    for x, weight in calls:
        representatives.setdefault((len(x), *weight.shape), (x, weight))
    selected = {}
    rows = []
    for shape, (x, weight) in representatives.items():
        expected = linear(x, weight)
        reference = triton.testing.do_bench_cudagraph(partial(linear, x, weight)) * 1000
        best = reference
        for bm, bn, splits in product((16, 32), (64, 128), (1, 2, 4, 8)):
            config = dict(bm=bm, bn=bn, splits=splits)
            fn = partial(projection, x, weight, **config)
            actual = fn()
            rrms = error(actual, expected)
            row = dict(
                shape=shape,
                config=config,
                rrms=rrms,
                passed=bool(actual.isfinite().all()) and rrms < 0.003,
                reference_hot_us=reference,
            )
            if row["passed"]:
                row["hot_us"] = triton.testing.do_bench_cudagraph(fn) * 1000
                if row["hot_us"] < best * 0.98:
                    best = row["hot_us"]
                    selected[shape] = config
            rows.append(row)
            output.write_text(json.dumps(dict(configurations=rows), indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    shape=shape,
                    reference_hot_us=reference,
                    best_hot_us=best,
                    selected=selected.get(shape),
                )
            ),
            flush=True,
        )
    checks = []
    for index, (x, weight) in enumerate(calls):
        config = selected.get((len(x), *weight.shape))
        if config is None:
            continue
        expected, actual = linear(x, weight), projection(x, weight, **config)
        rrms = error(actual, expected)
        checks.append(
            dict(
                index=index,
                shape=(len(x), *weight.shape),
                rrms=rrms,
                passed=bool(actual.isfinite().all()) and rrms < 0.003,
            )
        )
    passed = all(row["passed"] for row in checks)
    result = dict(
        configurations=rows,
        checks=checks,
        passed=passed,
        selected=[dict(shape=shape, config=config) for shape, config in selected.items()],
    )
    if passed:

        def run(candidate):
            for x, weight in calls:
                config = selected.get((len(x), *weight.shape)) if candidate else None
                if config is None:
                    linear(x, weight)
                else:
                    projection(x, weight, **config)

        # Cycle every actual weight in model order; this avoids extrapolating
        # from repeatedly reading one matrix from L2. These are projections
        # only, not an end-to-end decode graph or service throughput.
        result["all_projection_baseline_ms"] = triton.testing.do_bench_cudagraph(
            partial(run, False)
        )
        result["all_projection_candidate_ms"] = triton.testing.do_bench_cudagraph(
            partial(run, True)
        )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("configurations", "checks")}
        ),
        flush=True,
    )
    assert passed, "Selected projection changed a real layer beyond the fixed operator bound"
