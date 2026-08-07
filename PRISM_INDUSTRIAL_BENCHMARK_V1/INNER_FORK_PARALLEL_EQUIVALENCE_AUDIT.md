# Ordered fork parallelism equivalence audit

Date: 2026-08-07 (Asia/Shanghai)

The Metro W implementation now uses Linux fork pools for independent fold
preparation and registered candidate fits. Results are collected in registration
order; the mathematical implementation, FP64 dtype, row caps, folds, and
selection rules are unchanged.

Checks completed before formal continuation:

- Targeted Metro audit tests: 20 passed.
- Full frozen M1 regression suite: 68 passed.
- Real Metro serial (`PRISM_V211_W_INNER_WORKERS=1`) versus fork-parallel
  (`PRISM_V211_W_INNER_WORKERS=15`) comparison: both views had exact JSON
  equality after excluding only `elapsed_seconds` and the worker-count field.
- Materialized artifact SHA256 matches:

  - `full_sensor_secondary/PHYSICAL_OOF.parquet`:
    `8fa45480bcd88cfadfbce45a09f7564200fb724c337dc2bc4455171aeae58fa5`
  - `full_sensor_secondary/validation.parquet`:
    `e020caf75ca7bbcdd4857ceba42b73b1c8ef661a00a477189c0e3839345ca076`
  - `proxy_excluded/PHYSICAL_OOF.parquet`:
    `4721c0fcb3925925102b6f0d047683dc30d19aed2a07d07d5e662d416ba1cf68`
  - `proxy_excluded/validation.parquet`:
    `ecab9a641b38074d07acbeb7843085f34d6c1cd53e807ce8f8eb175c067e1cfe`

The resource smoke run showed two outer views, four forked fold workers per
view, and fifteen forked candidate workers per view without OOM. Test and OOD
data remained locked and were not accessed.
