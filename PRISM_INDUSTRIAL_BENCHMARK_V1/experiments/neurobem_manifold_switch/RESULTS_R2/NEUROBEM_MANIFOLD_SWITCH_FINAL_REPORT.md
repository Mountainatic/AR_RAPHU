# NeuroBEM manifold-aware PRISM switching — final report

## Decision

The registered hypothesis `t_alarm < t_diverge_static` was **not supported**. On the 12 frozen test trajectories, both PF_KCW and J_KCW diverged on 12/12 trajectories. No registered detector emitted a test alarm, so no known-model switch or causal local re-identification was activated.

This is a negative result. The evidence is consistent with recursive predictor instability that is not causally preceded by the registered residual/projection/tangent monitor. It does not establish a true inter-manifold transition, and it does not justify a universal OOD claim.

## Frozen provenance

- Generating commit: `0775fa819957db9679f5fbcfb3f97f9cc7b32107`
- PRISM core/model families and W family were unchanged; train-only parameters were refit under the same registered route family.
- Each CSV is a distinct entity; history never crosses a trajectory boundary.
- R2 uses 69 earliest parent flights (175 segments) for fit and 23 latest parent flights (60 segments) for calibration.
- All 12 test content hashes were excluded from fit/calibration. One train alias identical to `random_points.csv` was removed.
- R1 is retained but invalidated because 11 validation aliases duplicated 11 test files and one train alias duplicated the remaining test file.
- Formal test was accessed once after code/config freeze and was not used for tuning.

## Results

| Route | Static divergence | Median first divergence | Detector alarms | Switches | New models |
|---|---:|---:|---:|---:|---:|
| PF_KCW | 12/12 | 347.0 steps | 0 | 0 | 0 |
| J_KCW | 12/12 | 380.0 steps | 0 | 0 | 0 |

All six registered ablations had divergence rate 1.0 for both routes. Because alarms were absent, the switching and re-identification ablations were behaviorally identical to the static route on test.

The static divergence sensitivity rates at multipliers 0.8/1.0/1.2 were recorded in the machine-readable summary; this sensitivity analysis changed only the diagnostic threshold, never the model or test selection.

## Interpretation boundary

Teacher-forced one-step errors remain small on representative trajectories while free rollout explodes, separating local one-step prediction from recursive stability. The registered manifold signals did not reliably anticipate the explosion during development or test. Consequently the experiment cannot distinguish same-manifold support exit from a true regime transition; the safe classification is `INTER_MANIFOLD_TRANSITION_NOT_ESTABLISHED`.

No clipping, spectral constraint, Lyapunov penalty, threshold retuning, or test-driven model change was introduced.
