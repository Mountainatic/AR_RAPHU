# Post-Experiment Native-Support Maintenance

```text
historical_experiment_branch = prism-v2-1-1-metro-p60-joint-stability-final
historical_experiment_commit = 76231f3959c15183fbc781eb238034085ee71fc1
historical_protocol = HEAD_LEVEL_COMMON_SUPPORT
new_prospective_protocol = NATIVE_K_COMMON_ASSEMBLY_R1
maintenance_class = POST_EXPERIMENT_PROSPECTIVE_IMPLEMENTATION_MAINTENANCE
historical_results_recomputed = false
test_ood_reaccessed = false
evidence_reclassification = NONE
```

This commit changes only prospective execution after C1 is rebuilt. It must
never be recorded as the generating commit for the completed Metro-P60 M0--M8
evidence. That experiment remains:

> VALID UNDER HEAD-LEVEL COMMON-SUPPORT PROTOCOL, NOT NATIVE-SUPPORT OPTIMAL.

## Prospective support flow

```text
C1 maximal anchor universe
  -> registered anchor temporal folds
  -> candidate/profile-native K fitting (mask before cap)
  -> channel-local common-support K scoring
  -> selected K native-support refit
  -> active selected channels
  -> C assembly common support
  -> common-support best-K replay
  -> C -> W -> A / Joint
  -> route-aware final materialization on frozen assembly support
```

Old C1 parquet files cannot provide native support because their early anchors
were physically omitted. Prospective v2.1.1 code therefore requires the
`causal_history_floor`, `anchor_history_steps`, and
`sample_support_contract=NATIVE_K_COMMON_ASSEMBLY_R1` columns. Missing metadata
raises `NATIVE_SUPPORT_REQUIRES_REBUILT_C1_ANCHOR_UNIVERSE`; the implementation
does not infer missing rows from legacy `dependency_start` or `lmax_steps`.

## Static maintenance review

- C1 anchor origin generation depends on split/continuous-interval legality,
  purge/left buffer, W0, horizon, target window, and availability delay only.
- `base_origin_id` and `view_sample_id` identity formulas are unchanged for an
  origin that already existed.
- K native masks precede deterministic row caps.
- Every directly compared K set uses its maximum required history only for its
  local evaluation intersection; candidate fitting remains native.
- Exact-zero and nonzero K candidates record identical local scoring hashes.
- K results explicitly forbid raw cross-channel loss comparability.
- C registers folds on anchor rows, masks within each fold, and recomputes every
  active selected K on assembly common support before choosing best-K.
- W uses the C assembly mask before its fold/final caps; A inherits W rows and
  retains its causal maturity/latest-index logic.
- Corrected original Joint T_i -> V_i folds are retained, with the assembly mask
  applied inside each registered anchor fold. K representation changes columns,
  never rows.
- Final materialization applies frozen assembly support to train, validation,
  test, and OOD anchor frames before selected-K feature extraction.
- No historical result, freeze, final report, prediction, or bundle path is
  written by this maintenance.
- This review is static only. No runtime PASS is claimed.

## Execution record for this maintenance

```text
pytest_run = false
experiment_run = false
C1_rebuilt = false
M2_M8_rerun = false
test_ood_reaccessed = false
historical_results_modified = false
```
