# PRISM V1 implementation traceability

Status values are `NOT_STARTED`, `IMPLEMENTED`, `TESTED`, `RUN`, `PASS`,
`FAILED`, or `BLOCKED`.

| Contract | Governing clause | Code owner | Required test | Required artifact | Status |
|---|---|---|---|---|---|
| Five-dataset source/hash/license freeze | Master 3, 12; CPU 2, C0 | `src/prism_benchmark/stage0.py` | `test_stage0.py` | `dataset_registry/*` | PASS |
| Cadence and run/profile/month boundaries | Master 3, 9; CPU 2 | `stage0.py` | `test_stage0.py` | `CADENCE_AUDIT.json`, `RUN_BOUNDARIES.csv` | PASS |
| Exact task definitions and target change | Master 4; Theory 1.5 | `src/prism_benchmark/c1_contracts.py` | `test_c1_contracts.py` | `TASK_REGISTRY.json`, `targets/` | TESTED |
| Immutable split/sample IDs and purge | Master 9, 10; Theory 2.6, 8 | `src/prism_benchmark/c1_builder.py` | `test_c1_contracts.py` | `SPLIT_REGISTRY.json`, `SAMPLE_ID_REGISTRY.json` | TESTED |
| Input-only vs dynamic views | Master 5 | `src/prism_benchmark/c1_builder.py` | view ID and proxy isolation tests | `dataset_views/` | TESTED |
| Mean/Persistence/Seasonal/Trend | CPU 4.1, C2 | C2 baselines | analytic fixtures | `SIMPLE_BASELINES.csv` | BLOCKED |
| Ridge/PLS/DPLS/SVR/XGBoost | CPU 4.2, C2 | C2 soft sensor | leakage/budget/one-SE | `CLASSICAL_SOFT_SENSOR.csv` | BLOCKED |
| AR/ARX/NARX/N4SID/Hammerstein/HW | CPU 4.3, C3 | C3 sysid | nestedness/stability | `SYSTEM_IDENTIFICATION.csv` | BLOCKED |
| Per-channel profile and scale-matched AR | Master 7; CPU 5--6, C4 | C4 profile audit | profile isolation/sample equality | `PRISM_PROFILE_AUDIT.csv`, `AR_PROFILES/` | BLOCKED |
| Exact nested Urysohn ladder | Theory M3--M8, N4; CPU 5.2 | C4 Urysohn | exact nesting/rank/FP64 | `KERNELS/`, `NUMERICAL_CERTIFICATES/` | BLOCKED |
| Physics-First mature OOF residual | Theory P2--P7; CPU 5.3, C5 | C5 route I | rolling OOF/maturity/frozen-K/zero | `PRISM_MODELS.csv`, predictions | BLOCKED |
| True K-Joint AR | Theory J1--J4; CPU 5.4, C5 | C5 route II | joint-gradient/nested-zero/head isolation | `PRISM_MODELS.csv`, predictions | BLOCKED |
| Paired block bootstrap and Holm | Master 11; CPU C6 | C6 statistics | paired IDs/block semantics | `BOOTSTRAP/`, `CPU_FINAL_REPORT.md` | BLOCKED |
| Raw-data exclusion and round-trip bundle | Master 13; CPU 11--12 | packager | manifest/hash/privacy round trip | return bundle | BLOCKED |
