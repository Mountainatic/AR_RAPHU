# Formal Test Master Report

## Status

The one-shot formal test is complete and audited for 9 frozen views. No fit or selection call was made after opening formal test. All 43 saved model rows passed finite-value and prediction-hash verification. Frozen ZERO routes replayed exactly.

PMSM formal membership required a mechanical common-support restriction using the already frozen maximum input history. This restriction used only sample metadata (`origin`, `latest_available_target_index`, and `causal_history_floor`) and did not inspect target values, predictions, errors, or difficulty.

## Formal Findings

- Physics-first final models with positive persistence skill: 8/9 views.
- Development-active stages with positive formal incremental RMSE gain: 15/17.
- Boundary-active development stages (`|margin| < 1e-3`): 6.
- TEP maturity-5 Joint: `NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT`.
- AR/ARX/Linear-NARX formal baselines: `NOT_APPLICABLE_MISSING_PRE_FROZEN_DEVELOPMENT_CONTRACT`.

Highest persistence skill views: MetroPT P60, Debutanizer, PMSM proxy-excluded.

Lowest persistence skill views: MetroPT Oil20, TEP maturity-5, TEP input-only.

For TEP, delta metrics and level/persistence metrics must be read together. Strong level scores can be largely inherited from the persistence anchor when delta R2 is near zero.

## Interpretation

Formal evidence supports task-dependent stage utility and exact rejection of unnecessary stages. It does not support unique physical-structure identification. The existing E3 result supports multiscale value on Debutanizer under equal search budget but not universally, so the cross-task multiscale claim remains partial.
