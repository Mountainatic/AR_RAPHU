# PRISM Strict Nested-OOF Finalization: Complete

## Status

- Formal readiness: `READY` before opening.
- Formal test: `COMPLETED_AND_AUDITED` under one frozen OPEN event.
- Frozen views: 9; saved formal model rows: 43.
- Formal fit calls: 0; formal selection calls: 0.
- Prediction audit failures: 0; exact-zero replay failures: 0.
- External W: EMPS official file split, `COMPLETED`.
- OOD partition values: not accessed.

## Protocol

- Model commit: `73d0bd4c69148c0cb91459f39f166acacf5856b6`.
- Reporting commit: `d1090d4123daf63d3d31a9653d60155376f2b688`.
- Formal partition protocol lock SHA-256: `23f8b0290585160a8b2890db1abbce85e36f6981d3b75f0426feb5011bb5b792`.
- Formal opening: `2026-09-14T16:21:14.597515+00:00`.
- Fit policy: `NO_REFIT_REUSE_DEVELOPMENT_FINAL_SELECTED_CONTRACTS`.

PMSM required a mechanical common-support restriction after the initial readiness audit had checked only the task-level history floor. The repair used only frozen contract history and sample metadata; it did not use formal targets, predictions, errors, metrics, or difficulty. The original OPEN event and all failure/recovery records are preserved.

## Formal Outcome

- Physics-first final models with positive persistence skill: 8/9 views.
- Development-active stages with positive formal incremental gain: 15/17.
- TEP maturity-5 Joint: `NOT_APPLICABLE_MISSING_FROZEN_FEATURE_CONTRACT`.
- AR/ARX/Linear-NARX formal baselines: `NOT_APPLICABLE_MISSING_PRE_FROZEN_DEVELOPMENT_CONTRACT`.

## External W Outcome

- Dataset: EMPS, official `DATA_EMPS.mat` development / `DATA_EMPS_PULSES.mat` test split.
- C route: `ZERO_IDENTITY`; W route: `ZERO_IDENTITY`.
- KC RMSE: `1.8180981731e-06`; KCW RMSE: `1.8180981731e-06`.
- Development W margin: `0.02681374%`.
- Test W gain: `0.00000000%`; residual skill: `0.00000000%`.
- Verdict: `W_CORRECTLY_REJECTED_ON_THIS_TASK`.

The external result is a valid rejection result: W was not admitted under the frozen strict numerical rule, so KCW exactly replays KC on test.

## Claim Calibration

- A, strict nested-OOF rejects unnecessary stages: `SUPPORTED`.
- B, stage utility is task-dependent: `SUPPORTED`.
- C, admission margin distinguishes boundary activity: `SUPPORTED`.
- D, heterogeneous formal generalization: `PARTIALLY_SUPPORTED`; 8/9 views beat persistence, while MetroPT Oil20 did not generalize.
- E, multiscale value beyond search budget: `PARTIALLY_SUPPORTED` and task-dependent.
- F, unique structural interpretation: `NOT_SUPPORTED`.

Use the wording: PRISM identifies admissible predictive structures, not necessarily unique physical structures.
