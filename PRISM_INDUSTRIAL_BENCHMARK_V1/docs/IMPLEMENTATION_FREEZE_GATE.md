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

Status: `C1_SEMANTICS_APPROVED` on 2026-08-01. There are no unresolved C1
semantics gates.

- `HALF_OPEN_V1`: current `[t-W0,t)`, future `[t+h,t+h+W)`, and strict target
  history ending at `t-1-D`.
- `ROUND_HALF_UP_V1`: `floor(seconds/cadence+0.5)`, with the approved minima and
  the pre-registered 10% unsupported threshold.
- `DEPENDENCY_INTERVAL_V1`: `[t-Lmax,t+h+W+D)` and 10-minute ceil-rounded extra
  buffer at boundaries.
- `SHIFTED_SOURCE_TARGET_V1`: published Debutanizer target is used without
  reversal; main delay 0 and sensitivity delay 10 steps. The sensitivity name is
  `DELAY_10_STEPS` unless the 6-minute cadence is independently authoritative.
- `SAME_ROW_TARGET_V1`: TEP target stays on the XMEAS(40) row; delay sensitivity
  changes only strict target/residual/state availability by 5 steps.

## C4/C5 model gate

Status: `FROZEN_BEFORE_PARAMETERIZED_MODEL_VALIDATION` on 2026-08-02.

All previously open items are frozen in `configs/cpu_model_freeze_v1.json` and
explained in `docs/C2_C5_NUMERICAL_FREEZE.md`. CLI overrides are forbidden.

C1 target, split, sample-ID, purge, proxy-isolation and scaler validation is
`PASS`. Parameterized C2--C5 model implementation and fitting are authorized
under the frozen configuration.
