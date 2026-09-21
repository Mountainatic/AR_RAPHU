# PRISM Strict Nested-OOF Finalization

## 2026-09-21 unified H/W and R² integration

The consolidated six-public-dataset plus private-CZ protocol is documented in
[`PRISM_INDUSTRIAL_BENCHMARK_V1/docs/UNIFIED_HW_R2_PROTOCOLS_20260921.md`](PRISM_INDUSTRIAL_BENCHMARK_V1/docs/UNIFIED_HW_R2_PROTOCOLS_20260921.md).
The corresponding machine-readable registry, executable index rules, metric
reporter and tests are in the same benchmark directory.  This integration is
based on the strict nested-OOF authority and retains the supplementary Tanks
evidence; private CZ raw data is intentionally excluded.

This directory publishes the reproducible code, frozen protocol, audited summary tables, figures, and external-W result for the 2026-09-15 strict nested-OOF finalization.

## Results

- Formal status: `COMPLETED_AND_AUDITED`
- Frozen views: 9
- Saved formal model rows: 43
- Formal fit calls after opening: 0
- Formal selection calls after opening: 0
- Prediction-audit failures: 0
- Exact-zero replay failures: 0
- Physics-first views with positive persistence skill: 8/9
- Development-active stages with positive formal gain: 15/17
- External EMPS W verdict: `W_CORRECTLY_REJECTED_ON_THIS_TASK`

The canonical detailed result tables are under `results/prism_strict_oof_finalization_20260915/FORMAL_TEST` and `TABLES_FOR_PAPER`.

## Reproducibility

The implementation is under `PRISM_INDUSTRIAL_BENCHMARK_V1/`. The frozen model commit is `73d0bd4c69148c0cb91459f39f166acacf5856b6`. The formal protocol lock SHA-256 is `23f8b0290585160a8b2890db1abbce85e36f6981d3b75f0426feb5011bb5b792`.

The release result directory includes source runner scripts, protocol/configuration files, formal tables, E1-E6 diagnostics, figures, provenance, and hash manifests. It intentionally excludes large binary prediction files and raw external data from the normal Git repository:

- `FORMAL_TEST/PREDICTIONS/*.parquet`
- `W_EXTERNAL_VALIDATION/TEST_PREDICTIONS.parquet`
- `W_EXTERNAL_VALIDATION/DATA/*`
- the 691 MB complete archive

Those files remain in the separately verified final package. The original complete ZIP SHA-256 is `80d2f447a83f166d1c4e1a6acfd6871e6242fae8cddb3e2667ecaafcf849618f`.

## Claims

PRISM identifies admissible predictive structures, not necessarily unique physical structures. Claims D and E are partial; claim F (unique structural interpretation) is not supported. Negative formal results, including MetroPT P60-W and MetroPT Oil20-A, are retained.
