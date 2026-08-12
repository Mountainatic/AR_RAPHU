# Track B published-evaluator extension

This extension changes evaluation semantics only. It reuses the immutable
Track B contracts frozen at generating commit
`dcc969fec0aa8fda67bd5dcf508513b8aedd4e5b`; no model, candidate,
hyperparameter, coefficient, or development decision is changed.

The evaluator follows official repository commit
`a8451a119b9096eda980f71b55dbb05012b8c47a`,
`scripts/dynamics_learning/lighting.py::eval_trajectory`:

- the velocity branch recursively predicts linear/angular velocity and injects
  measured attitude plus the published future control at each step;
- the attitude branch recursively predicts normalized attitude and injects
  measured linear/angular velocity plus the published future control;
- `H=20`, `T=60`, 100 Hz, `N-H-T` windows, and the 12 official test
  trajectories are retained.

The old fully recursive result remains
`FULLY_RECURSIVE_STRESS_TEST = NONFINITE_RECURSIVE_DIVERGENCE`.

## Training-contract audit

The official `unroll_step` uses the same complementary ground-truth injection
during U-step training. Frozen PRISM coefficients were fitted with one-step
numerically certified ridge and W was selected using U10 fully recursive
development rollout. Consequently the training contracts do not match. The
strongest allowed claim is therefore:

`EXACT_PUBLISHED_EVALUATOR_ON_FROZEN_PRISM`

and never `EXACT_FULL_PUBLISHED_PROTOCOL`.
