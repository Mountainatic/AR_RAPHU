# PRISM v2.2(beta) vs RTA-TCN — protocol-matched Debutanizer benchmark

Date: 2026-08-24

Status: **benchmark-branch evidence only**. This file does not modify or redefine the frozen `prism-v2-2-beta-ct` branch.

## 1. Purpose

This experiment tests PRISM v2.2(beta) against the author-published RTA-TCN implementation from Zheng et al., *Chemical Engineering Research and Design* 212 (2024), 168–184, using the exact Debutanizer file distributed with the authors' code.

The goal is not to claim global SOTA. The goal is to obtain a clean, audit-ready, same-data / same-history-budget comparison with a published temporal deep-learning baseline.

## 2. Upstream author implementation

Repository: `CHM00/RTA-TCN-and-JITL-RTA-TCN`

Author files:
- `DC/RTA-TCN-DC.py`
- `DC/Debutanizer_data.txt`

Upstream SHA256 captured by the reproduction workflow:
- RTA-TCN code: `445f74e5635378fd2c04e508c6679bb7d36bbb50f576f9ad9623e9b99a1ee36a`
- Debutanizer data: `2129462fa25627d7705fb04239e566b559ddfe5c0556368204f8800484533d3e`

Author protocol recovered from the public implementation:
- 7 process inputs; target is column 8.
- sequence length = 40.
- `TRAIN_SIZE = 1648 + 40 - 1 = 1687`.
- active implementation does **not** use historical target `y`; the target-history block is commented out.
- RTA-TCN channels = `[40,40,40]`.
- epochs = 240.
- batch size = 64.
- learning rate = 0.001.
- seed = 1024.
- CPU execution.

## 3. Parser audit

The distributed data file has no header: its first row is numeric data.

The author script uses:

```python
pd.read_csv('Debutanizer_Data.txt', sep='\\s+')
```

without `header=None`. Pandas therefore consumes the first numeric record as a header.

For scientific auditing we report two protocols.

### Protocol A — Exact-author parser

Reproduce the author parser exactly. On Linux only, copy `Debutanizer_data.txt` to `Debutanizer_Data.txt` to emulate the paper-reported Windows case-insensitive filename behavior. No model/training semantics are changed.

Effective numeric rows: 2393.
Training targets: 1648.
Test targets: 706.

### Protocol B — Corrected parser

Use the same upstream file but read with `header=None`, preserving all 2394 numeric rows. No RTA-TCN architecture/training setting is otherwise changed.

Numeric rows: 2394.
Training targets: 1648.
Test targets: 707.

Protocol B is the preferred scientific result. Protocol A is retained as a code-reproduction audit.

## 4. PRISM v2.2(beta) information contract

To match the author's information budget, PRISM receives only the same 40-sample process-input window and no target history.

Therefore:
- inputs: only `u1..u7`;
- no `y_t`, no previous `y`, no persistence branch;
- no state carry from before the legal 40-sample window;
- each CT state is initialized inside the 40-sample window.

Benchmark-local temporal basis (not a change to the frozen silicon profile):

- D lags: `[0,1,2,4,8,16,32]` samples.
- CT time constants for M/S: `[1,2,4,8,16,32]` samples.

Branches:

- `D`: 7 variables × 7 explicit lags = 49 features.
- `M`: increment-only CT multiresolution basis = 42 features.
- `S`: CT absolute slow-state basis = 42 features.

Late assembly:

\[
\hat y = w_D p_D + w_M p_M + w_S p_S,
\qquad w_i\ge 0,\quad w_D+w_M+w_S=1.
\]

No persistence anchor is legal in this benchmark because the author model does not use target history.

### Training-only model selection

Within the 1648 training targets:
- first 70%: branch fitting for regularization candidates;
- next 15%: select branch Ridge regularization;
- refit on first 85%;
- last 15%: fit nonnegative simplex `Gamma_CT` assembly;
- freeze choices;
- refit branch models on all 1648 training targets;
- evaluate final test once.

Ridge candidate grid:
`[1e-4, 1e-3, 1e-2, 1e-1, 1]` with normalized penalty `alpha = lambda * N_train`.

Assembly ridge: `1e-3`.

## 5. Numerical certificate — corrected parser

Training feature audits:

| Branch | Features | Rank | Condition number |
|---|---:|---:|---:|
| D | 49 | 49 | 78.58 |
| M | 42 | 42 | 496.05 |
| S | 42 | 42 | 8794.21 |

All branches are full rank and far below the provisional beta hard threshold `1e8`.

Selected PRISM branch regularization:
- D: `lambda = 0.1`
- M: `lambda = 1.0`
- S: `lambda = 0.01`

Selected `Gamma_CT` weights:
- D: `0.5324945`
- M: `0.3269499`
- S: `0.1405556`

This means the final predictor remains structurally decomposable: roughly 53% explicit-delay prediction, 33% multiresolution CT prediction, and 14% absolute CT-state prediction at the prediction level. These are model assembly weights, not physical causal shares.

## 6. Results

### Protocol A — Exact-author parser

| Model | RMSE | R2 |
|---|---:|---:|
| RTA-TCN, author code rerun | **0.1186292** | **0.5700563** |
| PRISM v2.2(beta) | 0.1200872 | 0.5594230 |

Under exact-author parser semantics, RTA-TCN is modestly better:
- RMSE advantage relative to PRISM: about 1.21%.
- R2 advantage: about 0.01063 absolute.

### Protocol B — Corrected parser (preferred scientific audit)

| Model | RMSE | R2 |
|---|---:|---:|
| PRISM v2.2(beta) | **0.1185653** | **0.5699263** |
| RTA-TCN, parser correction only | 0.1198074 | 0.5608681 |

Under corrected parser semantics, PRISM v2.2(beta) is modestly better:
- RMSE reduction versus corrected RTA-TCN: about **1.04%**.
- R2 advantage: about **0.00906** absolute.

The sign of the small advantage changes when the one-row parser issue is corrected. Therefore the scientifically justified conclusion is **performance parity / same tier**, not dominance.

## 7. Interpretation

This benchmark is deliberately difficult for PRISM v2.2(beta): unlike the silicon forecasting experiments, no measured target state or target history is available. The task is a pure dynamic soft-sensing mapping from 40 historical process-input samples to the current quality variable.

The main observation is:

1. PRISM v2.2(beta) is numerically healthy here; the result is not explained by rank deficiency or catastrophic conditioning.
2. With only D/M/S linear branch predictors and constrained late assembly, PRISM reaches essentially the same test level as the published RTA-TCN implementation under the same information budget.
3. RTA-TCN fits the training set far more tightly (corrected parser train R2 about 0.9965) but test R2 remains about 0.5609, indicating a large train-test/generalization gap under this chronological split.
4. PRISM's competitive test performance is achieved with an explicit delay/scale/state decomposition and without a nonlinear neural residual in this benchmark.
5. This does **not** establish Debutanizer SOTA. Recent papers report much higher values under other protocols; their splits/information contracts must be verified before direct ranking.

## 8. Scientific claim allowed from this experiment

Allowed:

> On the author-released Debutanizer data, under a matched 40-sample, seven-process-variable, no-target-history information budget and the same chronological train boundary, PRISM v2.2(beta) achieved test accuracy statistically/numerically in the same tier as the published RTA-TCN implementation. With the corrected headerless-data parser, PRISM slightly outperformed the rerun RTA-TCN (RMSE 0.1186 vs 0.1198; R2 0.5699 vs 0.5609), while the exact author parser reversed the small ordering.

Not allowed:
- "PRISM is Debutanizer SOTA."
- "PRISM beats all deep-learning soft sensors."
- "The Gamma_CT weights are physical causal contribution percentages."
- comparison to a different Debutanizer dataset/version as if it were this exact file.

## 9. Evidence files

- `RTA_TCN_EXACT_AUTHOR_RESULT.md`: CI rerun of the unmodified author training/model semantics.
- `RTA_TCN_CORRECTED_PARSER_RESULT.md`: same author model with only `header=None` added.
- this file: matched PRISM/RTA-TCN benchmark synthesis.

## 10. Isolation rule

All files in this benchmark experiment are evidence attached to the isolated `PRISM v2.2(beta)` branch family. They must not be used to redefine or reconstruct PRISM v2.1.1 unless the user explicitly requests v2.2(beta) evidence.
