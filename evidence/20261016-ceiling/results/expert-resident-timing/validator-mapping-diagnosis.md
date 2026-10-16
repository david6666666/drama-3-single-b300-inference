# Preserved first zero-debug gate: invalid physical-mapping assumption

The original frozen run stopped after11 checks at L0/left_only/eager/fc2 and performed no timing. Both native and resident finalized outputs had already matched the archived native routed output bitwise. The FC2 comparison reported721,653 differing bytes.

Source review identifies a validator-contract error: candidate_check reconstructs mapping/valid from the archived oracle, while the benchmark now generates native routing metadata live for each arm. The native API may choose a different within-expert physical permutation. A comparison of physical FC2 storage must use that arm's live expanded-to-permuted map, with independent token/expert ownership checks. Finalized output comparison already uses each arm's live map.

The first failure, source freeze, result and tensors remain unchanged. The v2 follow-up changes only this checker contract, requires mapping bounds and bijection, token identity and expert identity, and saves live mapping provenance. It must not infer a permutation from output values or relax bitwise acceptance. Until that follow-up passes, this source diagnosis is not a claim that all zero-debug arithmetic is validated.
