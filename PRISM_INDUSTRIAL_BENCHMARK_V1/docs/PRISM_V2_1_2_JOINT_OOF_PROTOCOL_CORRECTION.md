# PRISM v2.1.2 Joint OOF Protocol Correction

## Scope

This release is an implementation/protocol correction to Joint development
OOF support. It is not a performance-tuning release. All data partitions,
candidate families, numeric grids, one-SE rules, practical activation rules,
input-path thresholds, and numerical certificates remain frozen.

The PRISM v2.1.1 Metro-P60 failed-stop remains immutable evidence. v2.1.2 uses
a new branch and a new result namespace.

## Old development support

Let `(T_i, V_i)` be the original registered development inner folds. W writes
OOF predictions by concatenating `V_0, V_1, V_2, V_3`. The old Joint code then
treated earlier W evaluation chunks as a new fit pool:

\[
D_{J,i}^{fit}=\bigcup_{j<i}V_j,
\qquad
D_{J,i}^{eval}=V_i.
\]

It also omitted fold zero because no earlier OOF evaluation chunk existed.
This is a nested/meta-OOF estimator, not the estimator registered for C/W.

## Corrected development support

The corrected Joint calls `inner_folds(development_train)` directly and uses
all registered folds:

\[
D_{J,i}^{fit}=T_i,
\qquad
D_{J,i}^{eval}=V_i,
\qquad i=0,1,2,3.
\]

The frozen `joint_predictive_fit` cap is applied independently to each `T_i`;
the frozen `validation_selection_per_fold` cap is applied independently to
each `V_i`. Both caps retain their deterministic SHA256-of-`base_origin_id`
semantics.

## Fold-local estimator construction

For every original fold:

1. K features are built from `T_i` and scored on `V_i`, using only M2-frozen
   active K support/profile/family.
2. The C-routed latent is refit as `T_i -> V_i`; W `physical_oof` values are not
   used as seeds.
3. The W stage's frozen Joint basis family and knot count are retained, but
   mean, scale, knots, and support are constructed only from the `T_i` C latent.
4. A target state uses each row's `latest_available_target_index` through a
   train-partition-only `BaseAccessor`; causal maturity is unchanged.
5. `J_K`, `J_KW`, `J_KA`, and `J_KWA` are jointly solved and scored. AR-only
   remains diagnostic and selection-ineligible; K-zero routes remain forbidden.

The final estimator remains capped full development train to full validation.
Selection, scoring, materialization, prediction path, and contract remain bound
to the same candidate ID.

## Provenance audit

W `PHYSICAL_OOF.parquet` and C `SELECTED_OOF.parquet` are read only for fold-ID
auditing. Every fold records row counts before/after cap, ordered SHA256 hashes
for `base_origin_id` and `view_sample_id`, and origin bounds. Dynamic Joint IDs
are independently aligned by `base_origin_id` to the registered input-only view;
the input-only identifiers must then exactly match C and W. Any mismatch stops
with `STOP_JOINT_FOLD_PROTOCOL_MISMATCH`.

The result explicitly records:

```text
joint_fit_source = ORIGINAL_REGISTERED_INNER_TRAIN_SUPPORT
joint_evaluation_source = ORIGINAL_REGISTERED_INNER_VALIDATION_SUPPORT
nested_oof_training_used = false
w_physical_oof_used_as_training_pool = false
```

## Diagnostic terminology

The numeric input gate is unchanged. Its diagnostic classification now
distinguishes geometric variance collapse, coefficient collapse, numerical
failure, and an MSE-only preservation failure. If variance, coefficient, and
numerical checks pass but the frozen MSE ratio fails, the class is
`INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED`; this does not change the
candidate's formal gate result.

## Why the correction is required

The old estimator had materially less and differently distributed training
support than both the registered C/W inner estimators and the final Joint
materialization. Its development MSE therefore could not be compared directly
to PF/C/W gate evidence. The corrected estimator uses the same original fold
contract and aligns the development selector with the final estimator's support
semantics, while preserving every frozen scientific decision rule.
