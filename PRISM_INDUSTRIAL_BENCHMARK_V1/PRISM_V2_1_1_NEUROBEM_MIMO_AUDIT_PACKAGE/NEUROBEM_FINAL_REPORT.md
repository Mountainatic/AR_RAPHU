# PRISM v2.1.1 NeuroBEM MIMO Audit — Final Report

## Status and evidence identity

- Experiment status: `COMPLETED`
- Generating branch: `prism-v2-1-1-neurobem-mimo-audit`
- Generating commit: `6a3b10cb1067cdbfc9df47d275575a7b5d6af198`
- Configuration SHA256: `b6a976c7623dc4218f1ad976370f23517b2e5e8666dc3739e603f0b6da049e32`
- Canonical PRISM v2.1.1 theory SHA256: `72b88c151af4e26a994a68bf8919f4f241ec8cfafe42e5cd1be3549b7c443821`
- Processed NeuroBEM archive SHA256: `a3e2e83d156b48a95c346c4fd9b3ded0b62781331dc613f1cb9ab285be2b36a8`
- Test/OOD access: exactly once, after development freeze
- Model selection after test: `false`
- Full regression suite at the generating commit: `207 passed`

The official paper describes 96 flights, while the distributed `Flights.txt` and processed archive contain the same 95 unique parent-flight IDs. The executable registry therefore uses the 95 actually distributed parent flights and records the discrepancy instead of inventing a missing flight.

One train parent flight (`2021-02-18-16-43-54`, four segments) has an incompatible cadence of approximately 164 Hz rather than 400 Hz. The entire parent flight was prospectively excluded; it was not interpolated or split across partitions. The resulting executable split is 62 train, 19 validation, 13 locked test and 1 cadence-excluded parent flight.

## Isolation and target semantics

Each continuous processed CSV segment is an independent history entity. Parent-flight ID is the split group. No lagged input, residual state, fold, or row cap can cross a segment or parent-flight boundary. All predictors for row `t` use information available no later than `t-1`.

The four K inputs are centered squared motor speeds, used as thrust proxies. The four outputs are three generalized torque targets and one body-z generalized force target. The fourth target is not claimed to be uniquely identified propeller thrust.

## Development selection

### K — causal four-input/four-output FIR

K passed all four per-axis variance, coefficient, MSE-preservation and numerical gates. The selected history is 64 samples (160 ms at 400 Hz), the longest registered candidate. This boundary selection is reported as such; the grid was not expanded after seeing the result.

| History (samples) | Fold losses | Mean normalized loss |
|---:|---|---:|
| 4 | 0.494173, 0.673329, 0.515928, 0.657709 | 0.585284 |
| 8 | 0.460513, 0.643270, 0.490710, 0.626975 | 0.555367 |
| 12 | 0.437657, 0.615409, 0.462186, 0.592166 | 0.526854 |
| 20 | 0.413172, 0.581561, 0.430348, 0.550343 | 0.493856 |
| 32 | 0.379993, 0.531037, 0.384526, 0.487965 | 0.445880 |
| 64 | 0.351075, 0.484675, 0.332520, 0.428266 | 0.399134 |

The matched-support exact-zero fold losses are 0.893991, 1.117019, 0.914643 and 1.125203. K improves their mean by 60.59%, with positive improvement in 4/4 folds.

### W — nonlinear context correction

W selected `SIGNED_QUADRATIC_AERO_CONTEXT`. Its four fold losses are 0.320284, 0.435498, 0.288079 and 0.373591 (mean 0.354363), versus the identity route's mean 0.399134. The registered improvement is 11.22%, positive in 4/4 folds; identity equivalence also passed.

Airflow is not directly observed in the processed data. W is therefore evidence for a useful speed/rate-dependent nonlinear correction, not a causal identification of drag, wind, or a particular aerodynamic law.

### A — mature residual state

A selected residual lags `[1, 2, 4, 8, 12, 20]`. Its fold losses are 0.010404, 0.010890, 0.010465 and 0.012088 (mean 0.010962), versus the exact-zero mean 1.034536. The improvement is positive in 4/4 folds.

At 400 Hz and one-step prediction, this very large gain is consistent with short-term residual persistence, filtering and closed-loop memory. It must not be interpreted as identifying wind or vortex-ring state.

### ERA/Hankel MIMO realization

A stable order-6 realization was found from the frozen K Markov parameters, with spectral radius 0.938478. Orders 10, 12 and 16 were unstable. However, the order-6 realization's development mean loss is 1.483578, above the frozen K loss 1.115070 and the registered maximum 1.137372 (`1.02 × K`). Its status is therefore:

`MIMO_REALIZATION_STABLE_BUT_NOT_PREDICTIVELY_PRESERVED`

The realization was excluded before test and has no test/OOD metrics. This is an algebraically stable compression, not evidence for uniquely identified physical states or poles.

## Locked test results

Axis order is roll torque, pitch torque, yaw torque, body-z generalized force. The locked test contains 13 parent flights, 31 continuous segments and 255,781 evaluated rows.

| Route | Normalized MSE | Per-axis RMSE | Per-axis R² |
|---|---:|---|---|
| K | 0.463294 | 0.017998, 0.016813, 0.005764, 1.368068 | 0.582714, 0.537521, 0.075912, 0.950679 |
| KW | 0.424745 | 0.016405, 0.014900, 0.005876, 1.040299 | 0.653321, 0.636749, 0.039471, 0.971481 |
| KWA / PF_SELECTED | 0.004709 | 0.001597, 0.001498, 0.000646, 0.102355 | 0.996715, 0.996329, 0.988396, 0.999724 |

W improves pooled normalized MSE and the roll, pitch and force axes, but yaw R² is slightly worse than K alone. That axis-level exception is retained rather than hidden by the pooled score.

## Locked high-speed subset

The pre-registered high-speed subset uses speed at least 15 m/s and contains 4,930 rows.

| Route | Normalized MSE | Per-axis RMSE | Per-axis R² |
|---|---:|---|---|
| K | 0.497210 | 0.032253, 0.043468, 0.011477, 3.794080 | 0.737258, 0.675115, 0.151571, 0.447216 |
| KW | 0.363702 | 0.028153, 0.025819, 0.011347, 2.844451 | 0.799809, 0.885375, 0.170706, 0.689301 |
| KWA / PF_SELECTED | 0.004404 | 0.003405, 0.003775, 0.001312, 0.173777 | 0.997072, 0.997549, 0.988920, 0.998840 |

## Paired flight-segment evidence

The 500-replicate paired cluster bootstrap resampled complete continuous processed segments. Values below are candidate-minus-baseline normalized-MSE differences; negative favors the candidate.

| Contrast | Mean difference | Percentile 95% interval |
|---|---:|---|
| KW − K | -0.042742 | [-0.059167, -0.026981] |
| PF_SELECTED − K | -0.481610 | [-0.608779, -0.406004] |
| PF_SELECTED − KW | -0.442163 | [-0.599934, -0.361878] |

These are percentile cluster intervals, not Holm-adjusted p-values.

## Decision

The formal frozen routes are `K`, `KW`, `KWA`, and `PF_SELECTED`; `PF_SELECTED` is bound to KWA. The experiment supports a reproducible linear motor-to-generalized-force K path, a useful nonlinear W correction, and a highly predictive mature residual A path under this one-step protocol. It does not support a test-eligible ERA compression or causal claims about unobserved airflow, drag, wind, vortex states, physical poles, or uniquely identified thrust.

Official dataset references: [NeuroBEM project](https://rpg.ifi.uzh.ch/NeuroBEM.html), [official downloads](https://download.ifi.uzh.ch/rpg/NeuroBEM/), [paper](https://arxiv.org/abs/2106.08015).
