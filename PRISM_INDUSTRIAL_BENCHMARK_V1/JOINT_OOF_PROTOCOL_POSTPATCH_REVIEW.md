# Joint OOF Protocol Post-Patch Review

Date: 2026-08-08

Branch: `prism-v2-1-2-joint-oof-protocol-correction`

Base commit: `29ff09f88ca4d3424cb36e4d508db9e2d220f119`

## Decision

PASS. The implementation is ready for a development-only M0--M5 run. M6 may
run only if M5 passes every frozen development gate. M7, M8, test, and OOD are
outside this run.

## Required review

1. **No historical OOF-as-fit expression remains.** There is no
   `fit = oof[oof_fold < fold]` construction and no equivalent accumulation of
   prior validation chunks.
2. **`PHYSICAL_OOF.parquet` is audit-only.** Joint reads only
   `base_origin_id`, `view_sample_id`, and `oof_fold`; those rows are used only
   for cross-stage fold-ID and provenance checks. The result contract records
   `w_physical_oof_used_as_training_pool=false` and
   `nested_oof_training_used=false`.
3. **All Joint folds come from `inner_folds(train)`.** The corrected support is
   `T_i -> V_i` for every registered fold, including fold 0. No
   `sorted(...)[1:]` skip remains.
4. **Row caps are frozen and correctly scoped.** Each `T_i` is capped by
   `joint_predictive_fit`; each `V_i` is capped by
   `validation_selection_per_fold`. C and W provenance comparison uses the same
   deterministic cap and ordered IDs.
5. **W basis construction is fold-local.** `_fit_c_routed` produces the fold's
   train/evaluation latent seed. `joint_w_basis` fits mean, scale, knots, and
   support from the current `T_i` seed only. Future-fold latent values cannot
   change an earlier fold's basis.
6. **A state remains causally mature.** Inner Joint creates a train-only
   `BaseAccessor` and constructs both `T_i` and `V_i` target state from each
   row's `latest_available_target_index`. Global validation, test, and OOD
   targets are not used in inner selection.
7. **Fold 0 is retained.** With four registered folds, every candidate must
   have four losses, and the fold-provenance audit must contain indices
   `0,1,2,3`.
8. **No test/OOD access was introduced.** M0--M6 results explicitly retain
   `test_accessed=false` and `ood_accessed=false`; the formal launcher stops at
   M6 and does not invoke M7/M8.
9. **Candidate binding is unchanged and explicit.** The candidate set remains
   exactly `J_K`, `J_KW`, `J_KA`, `J_KWA`. The selected candidate, selected
   fold losses, selected contract, selected prediction path, and frozen
   hyperparameters share one candidate ID. W columns in J_KW/J_KWA remain one
   jointly solved block; no `kw_scalar` exists.
10. **No frozen threshold or hyperparameter changed.** The protected C, W, A,
    `cpu_data`, and assembly source files are byte-for-byte unchanged from the
    base commit. The v2.1.2 config differs from the v2.1.1 config only in
    `protocol_id`, `output_root`, and `recommended_branch`. In particular,
    maximum MSE ratio remains 1.02, practical activation remains 0.10, the
    positive-fold rule remains unchanged, and all candidate/ridge/penalty grids
    are unchanged.

## Provenance audit semantics

Metro dynamic and input-only views share ordered `base_origin_id` support but
have intentionally distinct `view_sample_id` namespaces. The audit therefore:

- proves corrected dynamic Joint `T_i` and `V_i` have exactly the registered
  input-only base origins after the same cap;
- proves input-only registered `V_i` exactly equals both C and W OOF
  `(base_origin_id, view_sample_id)` rows;
- records SHA256 values for both namespaces, row counts before/after cap, and
  origin bounds.

Any mismatch produces `STOP_JOINT_FOLD_PROTOCOL_MISMATCH` before a candidate is
fit.

## Regression evidence before formal Metro run

- Python compilation: PASS.
- Frozen SRU/Metro regression suite plus v2.1.2 tests: `80 passed`.
- Dedicated v2.1.2 tests: `11 passed`.
- Serial/fork candidate evaluation: exact equality on the deterministic
  fixture.
- Git whitespace check: PASS.

No formal Metro v2.1.2 result is claimed by this review. It authorizes only the
development sequence requested by the correction protocol.
