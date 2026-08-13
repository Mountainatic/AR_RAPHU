# Trajectory path-excursion diagnostic protocol

The composition audit retained endpoint predictions but not intermediate
recursive states. This diagnostic therefore performs a pure deterministic
reproduction replay of the already accessed 12 formal-test trajectories. It
uses the identical frozen adapters, resampling operators, fixed-time histories,
test identities, physical horizons, and composition anchor grid. No fitting,
selection, tuning, or new test decision occurs.

For every path and channel it records terminal, maximum, and RMS error; first
crossing time; crossing and reentry counts; fraction inside the envelope; and
maximum/terminal ratio. The envelope is not redefined: each route/rate uses the
unchanged velocity, attitude, and body-rate bounds stored in the sampling-rate
calibration freeze. Full-state inside means all three registered channel bounds
pass simultaneously; no synthetic scalar full-state bound is introduced.

The registered diagnostic is `supported` only for a strict higher-rate ordering
of path maximum or earlier first crossing. Nonuniform route/channel/horizon
evidence is reported as `mixed`; thresholds must not be adjusted after replay.
