# Sampling-rate scaling protocol

- Parent R3 reporting commit: `e2caedd63011f1465cb8054c8b5dd2195f440d8a`.
- Sampling rates: 100, 200, and native 400 Hz.
- History modes: fixed 20 steps and fixed 200 ms (20/40/80 steps).
- 100 Hz must exactly reproduce R3 before any higher-rate result is valid.
- The 100-Hz model is the frozen R2/R3 adapter. Higher-rate models reuse the
  frozen R2 train-parent partition, W family, ridge grid, row cap, routes, and
  estimator semantics. No test row enters fitting or calibration.
- 100/200 Hz use the same offline, left-labelled, left-closed bin mean as the
  published Track-B preprocessing (10 ms and 5 ms); 400 Hz is native.
- Reliability bounds are separately derived from each rate/history calibration
  representation using the frozen R3 quantile, multiplier, and 90% nested rule.
- The primary horizon grid is the exact R3 100-Hz grid scaled in physical time.
- The formal test is accessed once after configuration, rate-specific adapters,
  and calibration results are committed. No test-driven retuning is permitted.
- No PRISM core change, clipping, stabilization, spectral constraint, or
  Lyapunov claim is part of this audit.
