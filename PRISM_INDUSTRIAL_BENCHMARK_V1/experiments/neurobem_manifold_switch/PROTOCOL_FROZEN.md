# PRISM v2.1.1 NeuroBEM Manifold Switch R1 — frozen protocol

Status: `R1_INVALIDATED; R2_FROZEN_BEFORE_VALID_TEST_ACCESS`

Parent PRISM code commit: `eff340ffca58f150a8d9870e2c55361ec71ca08a`.
The global estimator is the previously frozen literature-aligned Track-B
contract. No PRISM estimator, route, coefficient, ridge grid, W family, or
state update is modified by this experiment.

Each CSV is one entity. Histories are created independently inside an entity
and never cross CSV boundaries. Released identities are taken verbatim from
the prior `TRACK_B_SPLIT_MANIFEST.json`. Its released 236/11/12 count differs
from the manuscript 67/17/12 count; the released identity set is used and the
discrepancy is retained rather than silently relabelled.

R1 monitor normalization used the released validation directory, later found
to duplicate 11/12 test files and therefore invalidated. R2 uses the earliest
75% of released train parent flights for fitting and the latest 25% for monitor
calibration. Any train file whose SHA256 matches a test file is excluded.
The
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

R1 results are audit evidence only. R2 keeps all numerical thresholds unchanged
and refits the same PRISM estimator family with the corrected train-only data
contract before any valid test access.
