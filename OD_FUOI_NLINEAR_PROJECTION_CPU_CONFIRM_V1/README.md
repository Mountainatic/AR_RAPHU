# OD-FUOI NLinear Projection CPU Confirm V1

CPU FP64 reference implementation of the frozen v3.2 experiment. It imports the
shared L6 benchmark and published CPU/GPU predictions, fits one full four-channel
Urysohn surface per outer direction with a single deterministic GCV smoothing
scale, then derives the linear, rank-1-linear, nonlinear and matured-residual
objects without retraining old models.

The raw Excel workbooks are never read or packaged. Run with `RUN_CPU_CONFIRM.sh`;
all stages checkpoint into `results/checkpoints/latest.json` and can be resumed
with `RESUME_CPU_CONFIRM.sh`.

The finite-band nonlinear continuation width is preregistered as one adjacent
training-knot span on each boundary (`c_rho=1`). The time-shift placebo is a
40-minute positive shift without circular wraparound. Both are frozen in the
protocol before any outer result is read.
