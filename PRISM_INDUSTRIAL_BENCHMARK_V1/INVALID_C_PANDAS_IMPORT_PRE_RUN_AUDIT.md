# Invalid pre-run audit: C pandas import failure

Date: 2026-08-07 (Asia/Shanghai)

The namespace
`results_prism_v2_1_1_metro_p60_w_audit_INVALID_C_PANDAS_IMPORT_PRE_RUN_20260807`
is an immutable invalid pre-run and is excluded from formal scientific evidence.

- Source HEAD before repair: `23675b2aa35da638659aee74d3a46cbd77229762`.
- M0 passed and the frozen M1 regression suite passed with 62 tests.
- M2 completed all 27 K jobs with status `PASS`; the active-channel counts were
  9 for `full_sensor_secondary` and 8 for `proxy_excluded`.
- Both C views ended as `SOLVER_FAILED_RETAINED` after fitting, because
  `src/prism_benchmark/v211_c.py` referenced `pd.concat` without importing
  pandas. The retained exception was `NameError: name 'pd' is not defined`.
- The aggregate `STOP_KC_INPUT_PATH_COLLAPSED` label was therefore an
  implementation-failure consequence and is not a scientific input-path result.
- M2 did not complete, the development protocol was not frozen, and M3--M8 were
  not run.
- Candidate test and OOD data were not accessed (`test_accessed=false`,
  `ood_accessed=false`).

The repair imports pandas, routes selected C OOF materialization through a
directly tested helper, and adds a regression test that concatenates multiple
OOF frames, writes Zstandard Parquet, reads it back, and checks exact frame
equality. Formal execution must restart from an empty default namespace.
