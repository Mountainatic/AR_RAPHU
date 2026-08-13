# Composition-consistency audit protocol

This prospective diagnostic reuses the five sampling-rate adapters frozen at
generating commit `97e68c662199e001f9f50214e39e93bd98ddfa37`. It does not fit
or modify any model.

The physical horizon grid is frozen to 10, 20, 50, 100, 200, and 500 ms. Up to
16 anchors per trajectory are deterministically sampled from a 500-ms physical
grid after 200 ms of available history. All rates start at the same native
400-Hz timestamp. The last history state at every rate is replaced by the same
native measured initial state, and terminal error is evaluated against the same
native measured endpoint. Historical state and motor paths use their already
frozen rate-specific sampling operator.

Primary errors are median endpoint norms over trajectory/anchor pairs. Q90 and
nonfinite fractions are retained. Full-state norm is the Euclidean norm of the
velocity norm, quaternion geodesic attitude error, and body-rate norm. Cross-rate
defects use the identical metric between terminal predictions. PF velocity and
Joint attitude/body-rate components are reported separately.

The committed R3 reproduction artifacts and all adapter SHA256 values must pass
before calibration. Formal test is accessed once only after this protocol and
the calibration result are committed. The four 164-Hz training segments remain
excluded. No clipping, multi-step retraining, stability penalty, or test tuning
is allowed.
