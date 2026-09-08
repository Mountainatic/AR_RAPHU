# Cascaded Tanks benchmark data

This directory preserves the Cascaded Tanks nonlinear system-identification
benchmark package used by the PRISM v2.1.1 W-stage experiment.

The benchmark contains two independent 1,024-sample records sampled every four
seconds:

- `uEst`, `yEst`: estimation/development record;
- `uVal`, `yVal`: validation/test record;
- `Ts`: sample period (4 seconds, stored in the first CSV row).

The experiment must use only the estimation columns for model and
hyperparameter selection. The validation columns are opened only after the
development configuration and bootstrap block length have been frozen.

## Layout

- `raw/CascadedTanksFiles.zip`: original benchmark archive.
- `raw/CascadedTanksSetup.JPG`: supplied setup photograph.
- `raw/TanksBenchmark.pdf`: supplied benchmark description.
- `extracted/CascadedTanksFiles/dataBenchmark.csv`: CSV data used by the Python
  experiment.
- `extracted/CascadedTanksFiles/dataBenchmark.mat`: original MATLAB data.
- `extracted/CascadedTanksFiles/PicsVideo/`: supplied setup photographs and
  overflow video.
- `SHA256SUMS.txt`: repository-side integrity hashes.

The supplied paper describes the weak Bernoulli-type tank dynamics, hard
saturation, stochastic upper-tank overflow, short estimation record, and
unknown initial state. Those characteristics make this dataset useful for
testing whether the stagewise W correction captures nonlinear residual
structure left by an upstream dynamic predictor.
