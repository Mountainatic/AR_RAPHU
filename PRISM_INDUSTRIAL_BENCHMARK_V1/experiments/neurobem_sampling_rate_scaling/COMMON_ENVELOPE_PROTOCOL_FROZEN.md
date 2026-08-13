# Common-envelope reliability audit

This diagnostic uses each route's frozen 100-Hz `hz100_h20` velocity,
attitude, and body-rate calibration bounds for all three sampling rates. It does
not modify or replace any rate-specific bound. Full-state inside means all three
channel bounds pass; no scalar full-state threshold is created.

Existing path-excursion outputs lack per-step errors, so exact first crossing,
reentry, fraction inside, and common-envelope periodic-resynchronization
reliability require a deterministic reproduction replay of the already accessed
test identities. Models, controls, anchors, sampling operators, and predictions
remain frozen. This is not a new test selection or tuning access.

Operational reliability exactly retains the prior nested rule: a trajectory is
reliable when at least 90% of its 12-second resynchronized rollout is inside all
three bounds, and a horizon passes when at least 90% of the 12 trajectories are
reliable. Candidate physical intervals are the pre-existing 10, 50, 100, 200,
500, and 1000 ms grid. No result-dependent threshold or candidate is added.
