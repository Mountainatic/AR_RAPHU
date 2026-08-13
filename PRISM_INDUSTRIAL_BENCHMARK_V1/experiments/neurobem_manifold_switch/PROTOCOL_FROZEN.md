# PRISM v2.1.1 NeuroBEM Manifold Switch R1 — frozen protocol

Status: `FROZEN_AFTER_VALIDATION_BEFORE_TEST_ACCESS`

Parent PRISM code commit: `eff340ffca58f150a8d9870e2c55361ec71ca08a`.
The global estimator is the previously frozen literature-aligned Track-B
contract. No PRISM estimator, route, coefficient, ridge grid, W family, or
state update is modified by this experiment.

Each CSV is one entity. Histories are created independently inside an entity
and never cross CSV boundaries. Released identities are taken verbatim from
the prior `TRACK_B_SPLIT_MANIFEST.json`. Its released 236/11/12 count differs
from the manuscript 67/17/12 count; the released identity set is used and the
discrepancy is retained rather than silently relabelled.

Monitor normalization and thresholds are fitted from validation only. The
registered score is an equal-weight combination of normalized one-step
innovation and a geometry score. Geometry is 0.7 projection residual plus 0.3
causal tangent drift. The 0.995 validation quantile, three-sample persistence,
and all other values are frozen in `configs/full.yaml`.

Free rollout is evaluated for up to 1200 samples at 100 Hz. Divergence occurs
after five consecutive samples exceeding any registered bound: velocity
15 m/s, attitude 1.5 rad, body rate 15 rad/s, or a non-finite state. Sensitivity
is reported at 0.8, 1.0, and 1.2 times these bounds.

An innovation alarm at step a becomes available only after observation a.
Therefore switching can begin no earlier than a+1. A switch may causally
re-synchronize the local state history from observations already available at
that time. Detector-only tracks never re-synchronize. Unknown-regime fitting
uses only the post-alarm buffer, and promotion uses the following causally
arriving validation buffer with a frozen 5% improvement margin.

Validation found no positive evidence for the primary hypothesis: both PF and
Joint static routes diverged on 11/11 validation trajectories; the combined
monitor alarmed on 1/11 and that alarm followed static divergence. No threshold
or hyperparameter was changed after this result. The formal test is a single
frozen-protocol evaluation, not an opportunity to select or repair the method.

