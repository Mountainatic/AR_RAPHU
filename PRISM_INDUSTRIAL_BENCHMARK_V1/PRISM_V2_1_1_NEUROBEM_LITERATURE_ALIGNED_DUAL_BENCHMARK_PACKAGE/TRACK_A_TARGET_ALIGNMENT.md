# Track A target and metric alignment

Track A predicts six body-frame quantities: `Fx`, `Fy`, `Fz` in N and `Mx`,
`My`, `Mz` in Nm.  The physical training target is reconstructed from the
official processed signals using the NeuroBEM rigid-body convention: body
force is `m * a_body`, and body torque is `J * angular_acceleration + omega ×
(J * omega)`, with `m=0.772 kg` and diagonal inertia
`[0.0025, 0.0021, 0.0043] kg m²`.

The public prediction archive has a separate, authoritative column contract.
Its first 29 columns reproduce the processed trajectory, columns 30--35 are
the published method's predicted total body force/torque, and columns 36--41
are target-minus-prediction residuals.  The metric-reproduction gate therefore
uses columns 30--35 as prediction and their sum with columns 36--41 as paired
ground truth.  It does not reinterpret these columns as a base model plus a
network residual.

The evaluator pools samples and computes axis MSE first. `Fxy`/`Mxy` are the
square root of the mean x/y MSE, `Fz`/`Mz` are single-axis RMSE, and `F`/`M`
are the square root of mean three-axis MSE. Failure to reproduce the paper
within the preregistered 1% relative tolerance downgrades the comparison to
`PUBLISHED_AGGREGATE_COMPARISON_ONLY`; it does not trigger metric tuning.

`BODY_Z_GENERALIZED_FORCE` from the earlier four-output audit is not reused as
published `Fz`.  Track A is a new six-output target contract.
