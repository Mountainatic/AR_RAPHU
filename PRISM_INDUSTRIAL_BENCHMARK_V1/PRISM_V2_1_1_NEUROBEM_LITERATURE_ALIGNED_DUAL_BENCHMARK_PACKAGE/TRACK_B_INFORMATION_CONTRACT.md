# Track B information contract

Track B uses the official release directory identities, resets time per CSV,
and applies the official `pandas.resample("0.01s").mean()` transformation.
Motor speed is multiplied by `1e-3`.  History is fixed at `H=20`, the published
training reference is `U=10`, and formal evaluation recursively rolls 60
samples (600 ms).

The learned state is `[v, unit quaternion, omega]`; position is omitted.
Registered future motor commands are available. After the first prediction,
all future velocity, attitude, and angular-velocity values in PRISM's history
are model predictions. Future measured state and future target residual are
never read. `A` is disabled.

The velocity branch predicts the six-dimensional increment of `[v, omega]`.
The attitude branch predicts an SO(3) rotation vector, applies the quaternion
exponential, composes it with the current quaternion, and normalizes. The
velocity metric is mean squared Euclidean error over six components. The
attitude metric matches the official evaluator's norm of quaternion log,
`2 atan2(||q_v||, q_w)`, with sign canonicalization.

The official decoupled evaluator teacher-forces the complementary measured
state: its velocity predictor receives future measured attitude, and its
attitude predictor receives future measured velocity/rate. PRISM intentionally
obeys the stricter frozen no-future-state contract. Published values remain a
useful protocol-level numerical reference, but this information mismatch
precludes an exact same-information-set ranking claim.
