# PRISM v2.1.2 Joint OOF Protocol Pre-Patch Review

Review baseline: `prism-v2-1-1-metro-p60-w-audit` at
`29ff09f88ca4d3424cb36e4d508db9e2d220f119`.

Scope reviewed in full:

- `src/prism_benchmark/v211_joint.py`
- `src/prism_benchmark/v211_w.py`
- `src/prism_benchmark/v211_c.py`
- `src/prism_benchmark/v211_a.py`
- `src/prism_benchmark/cpu_data.py`
- `src/prism_benchmark/v211_selection.py`
- `src/prism_benchmark/v211_assembly.py`
- `src/prism_benchmark/v211_metro_runner.py`
- `src/prism_benchmark/v211_metro_config.py`

## Finding

The v2.1.1 Joint development OOF implementation does not use the registered
raw inner-fold training support. It reads W `PHYSICAL_OOF.parquet`, drops the
first OOF fold, and fits each Joint fold on prior W evaluation chunks:

```text
OLD:
D_J,i^fit  = union_{j<i} V_j
D_J,i^eval = V_i

implemented as:
usable_folds = sorted(oof_fold)[1:]
fit = oof[oof_fold < fold]
evaluation = oof[oof_fold == fold]
```

The registered C and W protocol instead calls `inner_folds(train)` and applies
the frozen row caps to each original pair:

```text
EXPECTED:
D_J,i^fit  = T_i
D_J,i^eval = V_i
```

For Metro-P60, C and W use all four registered folds. C applies
`joint_physical_fit` to `T_i`; W applies `wiener_fit` to `T_i`; both apply
`validation_selection_per_fold` to `V_i`. `_cap` is deterministic because it
uses the registered `deterministic_subsample`, which orders the selected
indices after selecting by SHA256 of `base_origin_id`.

## Why OLD and EXPECTED are not comparable

`union_{j<i} V_j` is a strict meta-OOF support assembled from prior evaluation
chunks. It is neither the registered original fold training support `T_i` nor
the final Joint support. It is substantially smaller than `T_i`, omits the
first fold from scoring, and changes the estimator's training distribution.
The final materialization is correctly fit as capped full development train to
global validation. Therefore the old Joint OOF gate compares a meta-OOF
estimator with PF/C/W estimators selected on registered `T_i -> V_i` folds and
with a final Joint estimator trained on full development support. The resulting
MSE gate is not a like-for-like preservation comparison.

## Confirmed surrounding semantics

- `inner_folds` constructs four deterministic expanding folds for the single
  Metro entity, with dependency-boundary buffering.
- C and W fit/evaluation features use `fit_split="train"` and
  `evaluation_split="train"` for inner folds.
- W `build_w_design` derives mean, scale, knots, and support only from the
  supplied fit latent; its implementation is fold-local when given fold-local
  inputs.
- Joint W is a real basis block whose coefficients are solved together with K
  and A. No `kw_scalar` exists in the Joint solve.
- Joint candidates are exactly `J_K`, `J_KW`, `J_KA`, `J_KWA`; AR-only is
  diagnostic and selection-ineligible, and K-zero is rejected.
- Final Joint already uses capped full train to full validation and should not
  be changed.
- A target state uses `latest_available_target_index`. The old Joint creates a
  `BaseAccessor(..., "validation", ...)`; although index maturity limits the
  rows used, this opens train+validation base partitions. The corrected inner
  Joint must create a train-only accessor and use it for `T_i` and `V_i`.
- W `PHYSICAL_OOF.parquet` remains valid provenance and fold-ID evidence, but
  must not be used as a Joint training pool.
- The shared input-path gate correctly preserves its numeric PASS/FAIL rules,
  but currently labels any failure `INPUT_PATH_COLLAPSED`, including an
  MSE-only preservation failure. Diagnostic classification must be made more
  precise without changing the gate result.

## Patch boundary

The correction will change only Joint development fold support, fold-local
construction/provenance auditing, diagnostic terminology, v2.1.2 namespace
plumbing, documentation, and regression coverage. It will not change frozen
data splits, model families, candidate grids, one-SE rules, activation or gate
thresholds, numerical certificates, or test/OOD access rules. The v2.1.1
failed-stop branch, release, and results remain immutable evidence.
