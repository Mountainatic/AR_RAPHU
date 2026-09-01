# Public5 + CZ Neural-3 Level-R2 reporting release

## Evidence class

This release is a reporting-only reconstruction from the frozen three-seed
Neural-3 test predictions produced by the six-dataset run. It does not retrain
or reselect a model, rerun test inference, or access OOD targets.

The registered model target is a change:

```text
mean(y[t+h:t+h+W]) - mean(y[t-W0:t])
```

The reporting pass reconstructs the corresponding future level by adding each
frozen change prediction to the registered current-level anchor. MSE, RMSE, and
MAE must be identical in change and reconstructed-level representations. Level
R2 and delta R2 are both retained because their denominators differ.

## Published scope

- Public heads: `DEB_C4__H5__W1`, `PMSM_PM5__H600__W60`,
  `METRO_P60__H6__W1`, and `METRO_OIL20__H120__W12`.
- CZ head: `CZ_D20`, for both `Rod_1_to_Rod_2` and
  `Rod_2_to_Rod_1` directions.
- Information sets: `input_only` and `dynamic`.
- Models: `LSTM`, `TimeMixer`, and `iTransformer`.
- Split: `test` only.
- Support: the immutable common support stored in
  `freeze/NEURAL3_EXTENSION_COMMON_SUPPORT.json`.

The aggregate contains 36 model/view rows and 12 best-by-view rows: 24 public
rows plus 12 CZ rows.

## Reproduction command

Run from the repository root with the frozen prediction run and Public-All C1
root already available locally:

```bash
python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/level_r2_reporting.py \
  --run-root /path/to/PRISM_V211_CZ_NEURAL3_SIX_DATASET_20260817_R2 \
  --public-root /path/to/PRISM_V211_NATIVE_PUBLIC_ALL_20260815 \
  --output-root /path/to/release-output \
  --common-support-only \
  --scope public5 --scope cz \
  --split test \
  --model LSTM --model TimeMixer --model iTransformer \
  --target-head DEB_C4__H5__W1 \
  --target-head PMSM_PM5__H600__W60 \
  --target-head METRO_P60__H6__W1 \
  --target-head METRO_OIL20__H120__W12 \
  --target-head CZ_D20
```

On AutoDL, the GitHub push and Release upload use the official academic network
acceleration environment:

```bash
source /etc/network_turbo
```

## Release artifacts

- `SIX_DATASET_LEVEL_R2_METRICS.csv`: all 36 Neural-3 rows.
- `SIX_DATASET_LEVEL_R2_BEST_BY_VIEW.csv`: best model per frozen view.
- `LEVEL_R2_RECONSTRUCTION_AUDIT.json`: reconstruction identity and no-rerun
  audit.
- `NEURAL3_EXTENSION_COMMON_SUPPORT.json`: immutable common-support registry.
- `SIX_DATASET_TEST_OOD_ACCESS_AUDIT.json`: historical lockbox access audit.
- `source_test_results/`: the 36 source `TEST_RESULT.json` records only. Raw
  predictions, checkpoints, public raw data, and raw CZ data are excluded.
- `SHA256SUMS.txt`: release-asset checksums.

## Interpretation boundary

These results are valid legacy six-dataset Neural-3 evidence on frozen common
support. They must not be merged silently with the later Active3 TEP-nowcast
leaderboard because final-fit and runner versions differ. Any PRISM comparison
must identify the compared support and protocol version explicitly.
