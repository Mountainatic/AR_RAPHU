# R3 frozen protocol

- Parent R2 reporting commit: `9fc0e05a82518a8c892fa6c7d087dce9330a2391`.
- The frozen R2 train-only PRISM checkpoint is loaded; no estimator is refit.
- Calibration uses the same chronological parent-flight holdout as R2 and
  excludes every test-content SHA.
- Perturbation magnitudes are fixed fractions of calibration state scales.
- Expansion thresholds and reliability error bounds are derived on calibration
  only using the registered quantiles and multipliers in `calibration.yaml`.
- Test is run once after source, configuration, and calibration freeze commit.
- Ground truth in resynchronization and channel-attribution tracks is an
  explicit observation intervention. It is never used in the free-rollout
  baseline or perturbation/Jacobian calculation.
- No clipping, stabilization, spectral constraint, or test tuning is allowed.
