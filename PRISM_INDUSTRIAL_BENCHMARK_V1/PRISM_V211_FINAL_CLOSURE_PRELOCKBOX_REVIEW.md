# PRISM v2.1.1 Metro-P60 final-closure pre-lockbox review

Status: `PASS_BEFORE_CANONICAL_M6`

- Branch was created from `abf7edd162b9e4282789d178e1c4415035817a60`; the historical execution branch is untouched.
- Canonical metadata is `PRISM_V2_1_1` plus practice revision `PRISM_V211_JOINT_PREDICTIVE_STABILITY_PRACTICE_R1`.
- The standalone v2.2 theory is withdrawn and archived; its estimator-practice content is in canonical v2.1.1 section 11A.11.
- K/C/W/A and legacy Joint estimator files, data splits, row caps, candidate sets, grids, gates and thresholds are unchanged.
- The predictive-stability estimator body is unchanged apart from module/symbol/metadata naming; a historical callable alias and exact-output regression test are retained.
- M5 recomputation is a hard error. Canonical development evidence is created only by source-result preservation plus metadata wrappers whose source fields remain byte-for-byte JSON-equal.
- M7 dispatch uses `joint_estimator_semantics`; unknown or missing semantics hard-stop with `STOP_ESTIMATOR_SEMANTICS_UNBOUND`.
- Predictive eta is replayed only during final fit. Prediction uses frozen coefficients and does not reapply shrinkage.
- Both `CHANNEL_COMPRESSED` and `FULL_BASIS` final paths are explicit and use the corresponding frozen K block.
- M6 canonical candidate IDs bind view, route, K representation, numerical alpha, predictive eta, decision SHA and practice revision.
- `JOINT_SELECTED_VS_PF_SELECTED` is registered before lockbox as comparison family `JOINT_VS_PHYSICS_FIRST`.
- M7 preflight and target-history causality audits are implemented without loading test/OOD.
- Test/OOD first-access audit is written before loading either partition and preserves the first-access timestamp.
- M8 performs no reselection, separates current ranking from historical context, and writes the required primary comparison and evidence summary.
- Full regression: `191 passed, 1 skipped`; historical baseline was `179 passed, 1 skipped`.
- At review time: `test_accessed=false`, `ood_accessed=false`.
