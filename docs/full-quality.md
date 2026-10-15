# Full-corpus audio quality

First performance repetition at each concurrency. WER is corpus WER in percent;
SIM is the upstream WavLM speaker-similarity score. Every row includes the full
corpus; high-WER outliers are retained. These are measurements, not user-approved
quality thresholds.

These are the saved outputs from the same frozen `d6ed4f4` vLLM and `dcd20d3`
standalone runs in [the performance comparison](full-comparison.md).
Each runtime has nine complete evaluation groups (9792 scored waveforms).
The reporter checks evaluation counts, zero skipped samples, matching sample
IDs/text and identical ASR settings before producing this table.

| C | Samples/runtime | vLLM WER % | Standalone WER % | Δ WER pp | vLLM SIM | Standalone SIM | Δ SIM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1088 | 1.917 | 1.541 | -0.377 | 64.273 | 64.315 | +0.042 |
| 2 | 1088 | 2.311 | 1.984 | -0.327 | 63.841 | 64.236 | +0.395 |
| 4 | 1088 | 1.909 | 1.834 | -0.075 | 64.008 | 64.216 | +0.208 |
| 8 | 1088 | 1.817 | 1.909 | +0.092 | 64.202 | 63.895 | -0.307 |
| 16 | 1088 | 1.951 | 1.725 | -0.226 | 64.117 | 64.288 | +0.171 |
| 32 | 1088 | 1.901 | 1.901 | +0.000 | 64.199 | 64.087 | -0.112 |
| 64 | 1088 | 1.842 | 2.001 | +0.159 | 63.871 | 63.760 | -0.112 |

C64 was scored for all three repetitions:

- vLLM-Omni: median WER 1.917% (range 1.842–2.093%), median SIM 63.871 (range 63.840–63.933).
- Standalone: median WER 1.516% (range 1.474–2.001%), median SIM 64.081 (range 63.760–64.406).

ASR uses `openai/whisper-large-v3` at
`06f233fe06e710322aca913c1bc4249a0d71fce1`, English, greedy decoding and
256 maximum new tokens. WER normalization and WavLM similarity use the
SGLang-Omni Seed-TTS evaluation implementation at
`e9e016b83e1a5d92b4232f90427a44afac8d84b6`.
The similarity model revisions are recorded in
[quality-models.json](../evidence/20261015-b300/quality-models.json).
Generation is stochastic and scheduling changes random draws. These measurements
do not establish statistical quality superiority or bitwise equivalence.

All per-sample ASR transcripts, scores and coverage are retained in the compressed
quality JSON files under [the evidence directory](../evidence/20261015-b300).
There is no manual listening or multilingual quality claim.
