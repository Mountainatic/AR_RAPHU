# Cascaded Tanks E1-compatible stagewise W experiment

This experiment is a single-task extension of the PRISM v2.1.1 E1 stagewise
ablation protocol. Cascaded Tanks is not part of the repository's original
native dataset registry, so the extension is explicitly labeled
`E1_COMPATIBLE_SINGLE_TASK_EXTENSION` rather than a native E1 rerun.

## Protocol

- Estimation/development: `uEst`, `yEst` only.
- Frozen test: `uVal`, `yVal`.
- Prediction mode: direct 16-step prediction using past outputs (64 seconds at
  the four-second sample period).
- Development selection: four expanding-window folds.
- Minimum mean development K+W R²: 0.80.
- W activation gate: at least 1% mean development RMSE gain and at least 75%
  positive folds.
- E1 stage order: `K`, `K+C`, `K+C+DELTA_W`, `K+C+DELTA_W+A`.
- C and A have no separately registered route in this extension and are
  therefore reported as rejected identity stages.
- Bootstrap block length: selected only from estimation residual ACF.
- Inference statistics: 500 paired moving-block replicates and Holm correction
  over the three primary adjacent-stage comparisons.

## Frozen configuration and result

The development-selected K is a linear ARX model with two output lags, sixteen
input lags, ridge alpha `1e-6`, and horizon 16. W is the existing v2.1.1 natural
cubic correction with 12 knots.

| Stage | RMSE | MAE | R² |
|---|---:|---:|---:|
| K | 0.526545 | 0.358488 | 0.938120 |
| K+C | 0.526545 | 0.358488 | 0.938120 |
| K+C+DELTA_W | 0.392143 | 0.255963 | 0.965678 |
| K+C+DELTA_W+A | 0.392143 | 0.255963 | 0.965678 |

The admitted W stage reduced RMSE by 25.525%. With the development-frozen block
length of 14, its 500-replicate paired interval was 14.384% to 39.389%; the raw
two-sided tail probability was 0.003992 and the Holm-adjusted value was
0.011976. All four stages used the same ordered 993-row scoring support.

Because the validation record had already been evaluated during exploratory
work before the E1-compatible report was generated, this result is a
retrospective replication and not a new blind confirmatory claim.

## Reproduction

Run the development screen and frozen test first:

```bash
export PYTHONPATH="$PWD/PRISM_INDUSTRIAL_BENCHMARK_V1/src"
python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_cascaded_tanks_w_experiment.py \
  --data PRISM_INDUSTRIAL_BENCHMARK_V1/datasets/cascaded_tanks/extracted/CascadedTanksFiles/dataBenchmark.csv \
  --output-root /path/to/cascaded-tanks-w \
  --phase screen
python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/run_cascaded_tanks_w_experiment.py \
  --data PRISM_INDUSTRIAL_BENCHMARK_V1/datasets/cascaded_tanks/extracted/CascadedTanksFiles/dataBenchmark.csv \
  --output-root /path/to/cascaded-tanks-w \
  --phase test
```

Then construct the E1-compatible stagewise report in a separate destination:

```bash
python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/build_cascaded_tanks_e1_stagewise.py \
  --data PRISM_INDUSTRIAL_BENCHMARK_V1/datasets/cascaded_tanks/extracted/CascadedTanksFiles/dataBenchmark.csv \
  --source-root /path/to/cascaded-tanks-w \
  --output-root /path/to/cascaded-tanks-e1 \
  --phase freeze
python PRISM_INDUSTRIAL_BENCHMARK_V1/scripts/build_cascaded_tanks_e1_stagewise.py \
  --data PRISM_INDUSTRIAL_BENCHMARK_V1/datasets/cascaded_tanks/extracted/CascadedTanksFiles/dataBenchmark.csv \
  --source-root /path/to/cascaded-tanks-w \
  --output-root /path/to/cascaded-tanks-e1 \
  --phase report
```
