# V20 changes

## Fast KAN line

- Preserved exact Stage1TargetDelayKAN checkpoint/model semantics.
- Kept sequence-first unique response evaluation and exact depthwise convolution.
- Replaced nested homotopy by shared warmup plus independent pruning forks.
- Added cross-seed validation-only one-standard-error selection.
- Added multi-process same-GPU job pooling, telemetry and CUDA profiling.
- Added selected prediction/support/delay/function/contribution artifacts.

## Variational line

- Added a formal cubic B-spline distributed-lag model.
- Added sequence-first B-spline evaluation on unique time points.
- Added scale-normalized group lasso and second-difference roughness penalty.
- Added monotone restarted FISTA and explicit KKT/prox-gradient histories.
- Removed duplicate response solves from each outer delay iteration.
- Added seed-0 screen, five-seed formal validation selection and noisy reruns.
- Added coefficients, knots, roughness matrices, full q, function and contribution outputs.

## Engineering and audit

- Added 118-test regression suite.
- Added nonempty-CSV policy and atomic JSON/CSV writes.
- Added GPU worker pool with resume, per-job logs, DONE records and telemetry.
- Added final scientific comparison and strict packaging validation.
- Added PowerShell one-command full run and final result packaging.
