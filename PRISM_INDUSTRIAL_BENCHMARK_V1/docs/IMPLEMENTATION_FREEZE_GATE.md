# Implementation freeze gate

The source protocol is pre-registered at commit `8fe7b497`. This document lists
only choices that the source protocol does not numerically settle. They must be
frozen before the dependent stage starts and may not be selected from test
results.

## C0 metadata gate

- `SRU_LABEL_REPRESENTATION`: the file has 10,081 dense rows and both output
  columns change almost every row, while source literature describes 1-minute
  process samples and 30-minute quality measurements. We must establish whether
  the dense targets are interpolation, model-filled values, or true record-time
  labels before constructing 5-minute targets.
- `TEP_LICENSE`: the canonical source DOI and access terms must be recorded; raw
  data will not be redistributed regardless.
- `DEB_SRU_LICENSE`: the files are copyrighted book supplementary material.
  Register use/citation terms and prohibit raw redistribution.
- `PMSM_SPLIT_IDS`: choose complete profile IDs using only ID, duration and
  completeness, then freeze the exact lists before any target/model metric.
- `TEP_SPLIT_IDS`: freeze complete `(source_partition,faultNumber,simulationRun)`
  lists with nominal/disturbance stratification and an unseen-disturbance OOD
  allocation before any target/model metric.

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

Until these items are frozen, C0 inspection is allowed; C1 model-ready shared
data and all fitting are blocked.

