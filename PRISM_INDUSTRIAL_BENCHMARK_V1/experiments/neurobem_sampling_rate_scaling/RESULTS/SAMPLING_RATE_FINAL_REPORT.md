# NeuroBEM sampling-rate scaling audit

## Result

The registered physical-time scaling hypothesis was not supported. Under the primary fixed-time history comparison, PF_KCW reliable horizons were 20/20/20 steps, corresponding to 200/100/50 ms at 100/200/400 Hz. J_KCW produced 10/10/4 steps, or 100/50/10 ms.

This is a mixed registered Outcome B/D: reliable step count is approximately fixed for PF and through 200 Hz for Joint, while elapsed reliable time contracts as sampling rate increases; Joint contracts further to four reliable steps at 400 Hz. Higher sampling rate improves one-step error but does not improve recursive reliability. Therefore lower one-step error must not be interpreted as greater recursive stability.

Full free rollouts still diverged on 12/12 test trajectories for both routes at every rate/history condition. Median divergence in physical time remains several seconds and is not the same quantity as the conservative calibration-frozen reliable horizon.

## History and channel diagnostics

Fixed-step versus fixed-time history does not rescue the higher-rate reliable horizon. Scaling history from 20 to 40/80 samples at 200/400 Hz leaves the primary conclusion unchanged.

The R3 attribution persists. PF velocity-only recursion diverges on 12/12 trajectories at every rate. Joint velocity-only recursion remains finite on all 12, while Joint attitude-only/body-rate-only divergence is substantial and body-rate-only worsens to 11/12 at 400 Hz. The data therefore do not support the hypothesis that native 400 Hz disproportionately rescues Joint attitude/body-rate recursion.

## Provenance and scope

The frozen 100-Hz R3 adapter exactly reproduced its registered calibration and test baselines before higher-rate results were accepted. A source-rate audit found four 164-Hz segments belonging to one train-fit flight; per user authorization, they were excluded from every newly fitted scaling adapter. No interpolation or synthetic 400-Hz row was introduced. Calibration and all five rate-specific adapters were committed before the formal test was accessed once.

No PRISM core, route family, ridge grid, reliability threshold, clipping, spectral constraint, or stabilization was changed. This audit is not a Lyapunov or global stability proof.
