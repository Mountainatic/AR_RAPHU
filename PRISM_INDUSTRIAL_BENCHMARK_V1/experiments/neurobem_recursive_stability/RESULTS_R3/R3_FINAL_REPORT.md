# PRISM NeuroBEM R3 recursive stability audit

## Result

R3 exactly reproduced the frozen R2 baseline: PF_KCW and J_KCW both diverged on 12/12 test trajectories, with median first-divergence steps 347 and 380.

The primary registered expansion-event hypothesis was not supported. The calibration-only sustained-growth thresholds produced no `t_expansion` event before divergence on any test trajectory. This is Outcome C: the registered local expansion event is insufficient to anticipate visible divergence.

Local sensitivity is nevertheless mildly expansive in a descriptive sense: median newest-state-block `sigma_max(J)` is about 1.055 for both routes, the two finite-difference scales agree, and the three paired-rollout epsilon scales agree closely. These are empirical finite-time diagnostics, not a formal Lyapunov exponent or proof.

## Reliable open-loop horizon

Using bounds frozen from calibration one-step errors and the registered nested 90% criterion, PF_KCW has a reliable horizon of 20 steps (200 ms) and J_KCW 10 steps (100 ms). Test divergence is zero through N=20 for both routes, while reliability degrades at N=50/100 and free rollout diverges universally. This supports Outcome B: PRISM is useful as a short-horizon predictor in an observed loop, not as an autonomous simulator.

## Channel and component attribution

PF_KCW remains unstable when velocity alone is recursively fed back (12/12 divergence), so PF has a strong velocity-path instability. J_KCW is stable for velocity-only recursion (0/12), while body-rate-only and attitude-only recursion diverge on 10/12 and 7/12 trajectories; Joint instability is therefore concentrated in attitude/angular-rate feedback and their coupling.

KC-only is not a cure: PF_KC and J_KC both diverge on 12/12. Adding W delays median divergence (PF 220.5→347; Joint 343→380) but does not prevent it. No K/C/W parameter was changed or refit.

Position is absent from the Track-B 10-dimensional state and is correctly marked not applicable. Force/torque are not the recursive Track-B targets in this frozen experiment.

## Scope

No clipping, state saturation, spectral constraint, Lyapunov penalty, model retraining, or test-driven threshold change was performed. R3 does not establish global dynamical instability, universal OOD failure, or a manifold transition.
