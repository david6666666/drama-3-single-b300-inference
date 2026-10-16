# FC1 native contract correction — one-expert exact PASS

Source `95fcb0ef2662f2caf9bfb7fe6863dc3ad7805b1b`; layer0/expert205, eight real native routes. The original `expert-fc1-native` failure remains unchanged. The v2 artifact replaces one Prims-TS output-resource class in its own Python process; no installed source, checkpoint, production code or precision was changed.

The native 16x256b TMEM output pair contains rows separated by8, not adjacent rows. The corrected store keeps the original packed E4M3 converter and scatters its low/high bytes into native logical rows. Native dequant scale uses `rcp.approx(q)` after `q=448*rcp.approx(amax)`; replacing `amax*(1/448)` removes the measured low-bit scale discrepancy. Raw validation tensors are never rearranged.

`result.json`: **42/42 exact checks PASS**, including all8192 FC1 E4M3 bytes,64 FP32 scales, eager execution, CUDA Graph replay after output poisoning, read-only inputs/weights, other stage storage and CUDA RNG. Dequantized difference is exactly zero. CPU and GPU generated identical CUBIN `d9790e399abfd5b90889ed1b0f402740d3d284b929c5416874eefd2ea7191566`.

CPU AOT never initialized CUDA. Actual shared memory is71544 dynamic+2048 static=73592 bytes; no local/stack spill. Correction: the generated PTX160/48 role hints sum to53248, but that sum does not establish actual register allocation or occupancy. Final-machine-code audits of the subsequent native prototypes withdraw the one-CTA inference; see ../register-lowering-audit/report.md. This arithmetic gate establishes no measured residency.

There are no timing or service performance results. The subsequent broader experiment in `../expert-fc1-expanded` passed layer0's five full routing distributions but stopped on a declared synthetic zero-input scale mismatch (native0, candidate2.23214294e-15). Therefore this successful fixture does **not** establish a general native FC1 implementation; the zero/clamp branch still requires correction and the remaining layers are not yet executed.

Reproduce the original v2 gate from this directory with the standalone venv's `python launch.py`; `frozen-inputs.json` records all source/oracle/artifact hashes. `protocol.json` records exact commands/environment, fixed shapes and stop rules. `epilogue.patch`, native/candidate SASS and CPU/GPU compilation artifacts preserve the implementation provenance.
