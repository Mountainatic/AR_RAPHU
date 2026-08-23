# PRISM V1 implementation traceability

Status values are `NOT_STARTED`, `IMPLEMENTED`, `TESTED`, `RUN`, `PASS`,
`FAILED`, or `BLOCKED`.

| Contract | Governing clause | Code owner | Required test | Required artifact | Status |
|---|---|---|---|---|---|
| Five-dataset source/hash/license freeze | Master 3, 12; CPU 2, C0 | `src/prism_benchmark/stage0.py` | `test_stage0.py` | `dataset_registry/*` | PASS |
| Cadence and run/profile/month boundaries | Master 3, 9; CPU 2 | `stage0.py` | `test_stage0.py` | `CADENCE_AUDIT.json`, `RUN_BOUNDARIES.csv` | PASS |
| Exact task definitions and target change | Master 4; Theory 1.5 | `src/prism_benchmark/c1_contracts.py` | `test_c1_contracts.py` | `TASK_REGISTRY.json`, `targets/` | PASS |
| Immutable split/sample IDs and purge | Master 9, 10; Theory 2.6, 8 | `src/prism_benchmark/c1_builder.py` | unit plus independent package validator | `SPLIT_REGISTRY.json`, `SAMPLE_ID_REGISTRY.json` | PASS |
| Input-only vs dynamic views | Master 5 | `src/prism_benchmark/c1_builder.py` | view ID, proxy and sample-count isolation checks | `dataset_views/` | PASS |
| Mean/Persistence/Seasonal/Trend | CPU 4.1, C2 | `cpu_simple_baselines.py` | `test_cpu_simple_baselines.py` | `SIMPLE_BASELINES_DEVELOPMENT.csv` | TESTED |
| Ridge/PLS/DPLS/SVR/XGBoost | CPU 4.2, C2 | `c2_models.py` | leakage/budget/one-SE | `CLASSICAL_SOFT_SENSOR_DEVELOPMENT.csv` | TESTED |
| AR/ARX/NARX/N4SID/Hammerstein/HW | CPU 4.3, C3 | `c3_models.py` | nestedness/stability | `SYSTEM_IDENTIFICATION_DEVELOPMENT.csv` | TESTED |
| Unique realized state-profile candidates and paired-fold activation | CPU 4.3, 5.3--5.4 | `cpu_data.py`, `v2_selection.py`, `v21_selection.py` | `test_cpu_data_and_selection.py`, `test_v2_frozen_core.py`, `test_v21_selection.py` | A/Joint and system-identification development results | TESTED |
| Per-channel profile and scale-matched AR | Master 7; CPU 5--6, C4 | `c4_prism.py` | profile isolation/sample equality | `PRISM_PROFILE_AUDIT.csv`, joint predictions | TESTED |
| Exact nested Urysohn ladder | Theory M3--M8, N4; CPU 5.2 | `urysohn.py`, `c4_prism.py` | exact nesting/rank/FP64 | channel contracts and numerical certificates | TESTED |
| Physics-First mature OOF residual | Theory P2--P7; CPU 5.3, C5 | `c5_models.py` | rolling OOF/maturity/frozen-K/zero | OOF residuals and predictions | TESTED |
| True K-Joint AR | Theory J1--J4; CPU 5.4, C5 | `c5_models.py` | Urysohn subspace/nested-zero/head isolation | predictions and selection contract | TESTED |
| Paired block bootstrap and Holm | Master 11; CPU C6 | `c6_final.py` | paired IDs/block semantics | `BOOTSTRAP_PAIRED.csv`, `CPU_FINAL_REPORT.md` | TESTED |
| Raw-data exclusion and round-trip bundle | Master 13; CPU 11--12 | `build_cpu_bundle.py` | manifest/hash/privacy round trip | return bundle | TESTED |
