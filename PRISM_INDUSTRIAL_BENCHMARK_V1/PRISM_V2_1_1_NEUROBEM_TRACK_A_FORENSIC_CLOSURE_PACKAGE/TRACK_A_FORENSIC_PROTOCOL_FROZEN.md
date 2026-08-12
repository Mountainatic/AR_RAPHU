# PRISM v2.1.1 NeuroBEM Track A Forensic Closure R1

Status: `FROZEN_BEFORE_FORENSIC_RECOMPUTATION`.

This stage is read-only with respect to every PRISM and literature model. It
does not fit, select, retrain, stabilize, clip, or alter K/C/W/A. Historical
Track B divergence is retained and is outside this closure.

## Ground-truth contract

- mass: exactly 0.772 kg;
- inertia: exactly diag(0.0025, 0.0021, 0.0043) kg m^2;
- force: `m * measured body linear acceleration`, where the official processed
  acceleration includes gravity;
- torque: `J * measured body angular acceleration + omega cross (J omega)`;
- frame: NeuroBEM body front-left-up frame;
- timestamps: prediction and processed physical signals use the same released
  row with no shift, trim, interpolation, or added filter;
- quaternion source order is qx,qy,qz,qw and is not needed by this body-frame
  rigid-body target formula;
- released residual force/torque columns are forbidden as GT inputs.

Primary sources are the official NeuroBEM dataset README/code and the official
RCL-NUS/NeuroMHE `ground_truth.m` reproduction helper.

## Metric contracts

`RSS21_METRIC` computes component MSE over samples, averages the component MSE
inside a group, then takes the square root. Thus `F` divides the summed force
component squared error by `3N`, and `Fxy` by `2N`.

`NEUROMHE_METRIC` uses vector error. Thus `F` is the square root of the summed
three-component squared error divided by `N`, and `Fxy` uses `N`, not `2N`.
The six individual-axis RMSE values are identical under both contracts.

## Identity and reproduction rules

The official NeuroBEM `testset.txt` segments are mapped to the 13 NeuroMHE
Table-V trajectories using the official `.mat` oracle filenames and published
table order. This validates the NeuroMHE identity. RSS test identity is marked
unverified unless an RSS primary source explicitly binds its aggregate to the
same manifest; numerical proximity is never identity evidence.

The pre-frozen reproduction gate is relative difference <= 1%. Because the
published trajectory oracle contains values rounded to three decimals, an
additional absolute tolerance <= 0.0005 is allowed. The rule is an OR and was
frozen before recomputation.

Forensic sensitivity may report `SAMPLE_POOLED`, `FLIGHT_MACRO_AVERAGE`, and
`AXIS_RMS`. None may be selected merely for proximity to the RSS row.

## Decision rule

- `EXACT_DIRECT_COMPARISON_VALIDATED`: GT, identity, metric and RSS aggregate
  reproduction all pass.
- `PUBLISHED_AGGREGATE_COMPARISON_ONLY`: PRISM values remain valid and the
  published reference is usable, but released artifacts do not reproduce the
  RSS aggregate.
- `DIRECT_COMPARISON_NOT_ESTABLISHED`: target, identity, or metric semantics
  remain materially unresolved.
