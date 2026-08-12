# Track B exact published training protocol

This prospective experiment changes only the Track B training information and
unroll contract.  The PRISM K/C/W definitions, four formal routes, numerical
ridge grid, W candidate family, H=20, A-disabled status, and test metric are
unchanged.

The frozen source reference is
`arplaboratory/long-horizon-dynamics` commit
`a8451a119b9096eda980f71b55dbb05012b8c47a`, especially
`scripts/dynamics_learning/lighting.py::unroll_step` and
`scripts/hdf5.py::hdf5_trajectories`.

For velocity training, PRISM recursively uses its predicted linear and angular
velocity.  At each step it injects the corresponding measured attitude and the
published future motor command.  It never injects future measured velocity or
angular velocity.

For attitude training, PRISM recursively uses its predicted, normalized
quaternion.  At each step it injects the corresponding measured linear and
angular velocity and the published future motor command.  It never injects the
future measured attitude.

The training window is exactly `N-H-U`, with `H=20` and `U=10`.  The test
evaluator is exactly the already-audited published decoupled evaluator with
`T=60`.  Training and evaluation call the same state-update helper.

PRISM remains an FP64 numerically certified ridge estimator rather than a TCN.
Its block fits use a deterministic U=10 recursive roll-in followed by the same
closed-form ridge solve: K is fitted first, C is fitted on frozen K, W is fitted
on the frozen C latent, and Joint solves its registered columns together.  This
aligns the information/unroll contract without importing the official neural
architecture or adding a stabilizer.

The manuscript reports 67/17/12 trajectories, while the frozen official data
release contains 236 train CSV segments, 11 validation CSV segments, and 12
test trajectories.  This known discrepancy is retained.  Actual training and
validation identities are the official release directory members; the formal
test remains exactly the 12 official test trajectories.

Published TCN scores and earlier PRISM divergence were known before training.
Neither is used for selection.  Development selection uses only train and
validation.  No test data may be read until the generating commit is clean and
the development freeze is complete.
