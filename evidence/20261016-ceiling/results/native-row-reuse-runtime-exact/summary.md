# Preserved production-gate failure

Frozen runtime345c6a18;8 complete cases and1923 checks passed before the first byte mismatch. The next case had exact public outputs but a differing private padding GDN0 state in slot143. No performance or service work followed.

CPU reconstruction found unequal actual inputs: the reused originalP1536 graph retained previous prompt rows1367..1421 in its new padding domain;53 of those55 rows differ from the new candidate graph's EOS/zero initialization. `prepare` copies real prompts only and `row_metadata` does not overwrite frames. Only the saved GDN0 tensor was fully localized; later states were not checked after stopping.

The independentv2 binds identical full inputs, adds pre/post input equality and one original/original deterministic check, then repeats every original exact gate. No runtime change and no exclusion of padding state.
