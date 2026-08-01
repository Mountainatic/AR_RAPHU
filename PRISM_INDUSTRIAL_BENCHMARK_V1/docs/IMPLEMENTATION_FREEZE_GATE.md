# Implementation freeze gate

The source protocol is pre-registered at commit `8fe7b497`. This document lists
only choices that the source protocol does not numerically settle. They must be
frozen before the dependent stage starts and may not be selected from test
results.

## C0 metadata gate

Status: `C0_APPROVED` on 2026-08-01. There are no unresolved C0 gates.

- `TEP_SPLIT_IDS`: frozen as `TEP_RUN_FAULT_HOLDOUT_V1`; Training faults 0--15
  use runs 1--400 for train and 401--500 for validation, Testing faults 0--15
  form the locked main test, Testing faults 16--20 form the separately reported
  locked unseen-disturbance OOD set, and Training faults 16--20 are discarded.
- `PMSM_SPLIT_IDS`: frozen as `PRISM_PMSM_SPLIT_V1`; all complete profiles are
  ranked by `(row_count, profile_id)` into equal short/medium/long strata, then
  ordered by the approved SHA256 construction and allocated within each stratum
  by deterministic largest remainder at 60/20/20.

Resolved before model inspection:

- TEP: 180 s cadence; record-time main target; 900 s analyzer-maturity
  sensitivity for product analysis; canonical DOI `10.7910/DVN/6C3JR1`;
  raw redistribution prohibited by this project.
- Debutanizer: 360 s cadence; the distributed file's eight-sample target
  pretranslation is preserved in the record-time main view; 3600 s label-delay
  sensitivity is separate; raw redistribution prohibited.
- SRU: published Line 4 dense series at 60 s; `y1=H2S`, `y2=SO2`; the conflicting
  30-minute analyzer description remains documented as provenance sensitivity,
  not silently mixed into the primary record-time task.
- MetroPT-3: observed timestamp cadence is 10 s; the official PDF's four fault
  intervals are frozen as OOD masks; CC BY 4.0.
- PMSM: 0.5 s cadence and CC BY-SA 4.0.

## C1 target/data gate

- Exact half-open discrete endpoint convention for the current and future means.
- Exact deterministic rounding rule when physical horizons are not integer
  multiples of cadence (the protocol only specifies nearest causal integer and a
  10% failure threshold).
- Exact purge formula for each model family and inner fold.
- Exact handling of the Debutanizer file's already shifted target in the
  record-time main view and 60-minute label-delay sensitivity view.
- Exact TEP target maturity interpretation for `XMEAS(40)`.

## C4/C5 model gate

- Lag-basis family and its candidate dimensions for every channel class.
- Amplitude basis family, knot placement rule, and candidate `M_x` values.
- Full finite-Urysohn candidate `M_tau` values.
- Adaptive per-channel rank candidate set and rank tie-break order.
- Numerical grids for `lambda_0`, `lambda_tau`, and `lambda_x`.
- Exact one-SE complexity key across zero/linear/rank/full candidates.
- KKT, condition-number, Gram/Schur, HS-error and rank-margin thresholds.
- Exact solver/initialization/refit contract for rank-R and K-Joint AR.
- Exact residual-AR and Joint-AR state candidate families and orders.

C1 shared-data implementation is authorized. Model fitting remains blocked until
the C1 target, split, sample-ID, purge, proxy-isolation, and scaler artifacts pass
their tests and validation report.
