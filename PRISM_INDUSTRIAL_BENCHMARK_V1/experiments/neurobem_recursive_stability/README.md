# NeuroBEM R3 recursive stability audit

This experiment does not change PRISM. It loads the frozen R2 train-only
adapter and audits finite-time perturbation growth, the newest-state numerical
Jacobian block, declared periodic observation re-synchronization, and recursive
channel attribution.

The Track-B state contains linear velocity, quaternion attitude, and angular
rate. Position is not present and is reported as not applicable.

The numerical 9x9 Jacobian is only the newest-state block of the transition.
The predictor has a 20-step history, so finite-time paired rollouts are the
primary history-aware expansion diagnostic. Neither quantity is called a
formal Lyapunov exponent.
