# PRISM v2.1.1 NeuroBEM Multi-Horizon × Wiener-Prior Audit R1

Status: `FROZEN_BEFORE_NEW_MULTI_HORIZON_TEST_SCORE_ACCESS`.

This experiment is a `POST_LOCKBOX_PROSPECTIVE_EXTENSION` of the completed
`PRISM_V2_1_1_NEUROBEM_MIMO_AUDIT_R1`. The official NeuroBEM test partition was
already opened by that historical experiment. This extension therefore makes
no virgin-lockbox claim. Its prospective claim is narrower: all new horizons,
W arms, maturity mappings, baseline contracts, metrics, speed bins, statistics,
figures and reporting fields are frozen before any new multi-horizon test score
is computed. Test results cannot alter those choices.

The machine-readable authority is `MULTIHORIZON_CONFIG_FROZEN.json`.

## Questions

The experiment asks whether the very large h=1 A gain decays when direct
forecasting extends from 2.5 ms to 200 ms; whether canonical `W(q_K)` adds value
without measured motion context; whether an explicitly labelled aerodynamic
context extension adds value beyond that generic W; and whether the integrated
motor-to-generalized-force K topology remains stable across horizons.

Negative results are retained. W0, W1 and W2 are ablation arms and are never
globally winner-selected against one another.

## Causal time contract

For target row `t` and horizon `h`, the prediction origin is `t-h`. Motors,
body velocity, body angular velocity, target history, residual history,
scalers and knots may use information only at indices no later than the
prediction origin. No information in `t-h+1 ... t` is legal.

For A, registered ages are `[0]`, `[0,1,3]` and
`[0,1,3,7,11,19]` relative to the prediction origin. Thus target-row residual
lags are `h+age`; h=1 exactly reproduces the old `[1]`, `[1,2,4]` and
`[1,2,4,8,12,20]` mappings. Residuals are strictly segment-local and must
already be mature at the origin.

Each continuous processed segment is a history entity and each parent flight
is an indivisible split group. The historical 62/19/13 train/validation/test
parent split and whole-flight cadence exclusion are reused without change.

All routes within a horizon use identical target rows. Per-horizon metrics use
the horizon-native common route support. Horizon curves additionally use the
intersection of legal target rows across all six horizons.

## Frozen model families

K remains a four-input/four-output causal FIR in centered squared motor speed,
with histories `[4,8,12,20,32,64]`. Each horizon selects within that same grid;
the grid is not extended after results. K is fit on candidate-native support and
scored on the horizon's common route support.

W0 is identity only. W1 contains identity and train-knot natural cubic bases of
the four frozen K latent coordinates; it cannot read velocity, rate or speed.
W2 contains the entire W1 pool plus signed-quadratic and natural-cubic speed
context candidates using origin-causal body velocity, body rate and frozen K
prediction. W2 is an `AERODYNAMIC_CONTEXT_EXTENSION`, not canonical PRISM W.

Every W arm materializes both NO_A and WITH_A. A reads only mature W-arm
residual history and retains exact zero. No A-only route is registered.

External diagnostics are persistence, target-only multivariate linear AR and
linear NARX. They obey the same origin cutoff and are not PF candidates.

## Evaluation and statistics

All six horizons and all six PRISM routes enter the single formal extension
test access. Metrics are reported per axis and pooled, on native and
common-horizon support, the locked high-speed challenge (speed at origin at
least 15 m/s), and fixed speed bins `[0,5)`, `[5,10)`, `[10,15)`, `[15,+inf)`.
The high-speed set is not called OOD.

Primary paired uncertainty uses 5,000 parent-flight cluster bootstrap draws;
secondary sensitivity uses 2,000 continuous-segment cluster draws. All five
pre-registered contrasts are always reported. No p-values are generated.

Integrated K topology is computed by summing frozen FIR coefficients over lag.
Roll, pitch and body-z signs are compared with the pre-registered rotor geometry
templates. Yaw remains descriptive because no rotation-direction sign template
is registered.

## Lockbox sequence

All horizon development, W arms, A candidates and baselines finish first. A
single global freeze then binds contracts, config hash, source hash, reporting
schema and generating commit. The worktree must be clean. Only then may one
formal multi-horizon test materialization run. A second access must hard fail.

The old R1 result directory and artifacts remain immutable and are not copied,
overwritten or reclassified by this experiment.
