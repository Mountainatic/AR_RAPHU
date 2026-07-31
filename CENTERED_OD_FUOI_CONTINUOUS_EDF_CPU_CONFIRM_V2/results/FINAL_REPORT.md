# Centered OD-FUOI Continuous-EDF CPU Confirm V2 Final Report

- Estimator status: `SMOOTHING_SELECTION_UNRESOLVED`
- Model registration: `ESTIMATOR_UNRESOLVED`
- NLinear projection: `NLINEAR_PREDICTIVE_ONLY`
- Nonlinear increment: `NONLINEAR_INCREMENT_NOT_CERTIFIED`
- Residual: `RESIDUAL_EXACT_ZERO`

## Continuous EDF selection

- sheet1_to_sheet2: d_min=`7.423157`, d_1SE=`1.067359`, lambda_full=`4.510007600e+04`.
- sheet2_to_sheet1: d_min=`26.390740`, d_1SE=`17.744388`, lambda_full=`6.092594520e+00`.

## Frozen L6 results

| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |
|---|---:|---:|---:|
| Persistence | 0.314405304 | 0.658043636 | 0.513060810 |
| old K-only | 0.325154539 | 0.486323887 | 0.412382911 |
| NLinear-U | 0.217866830 | 0.472308647 | 0.365854318 |
| R1-LIN-DERIVED | 0.323299365 | 0.523017720 | 0.433205465 |
| LIN-UOI | 0.323421087 | 0.523427073 | 0.433494738 |
| FULL-UOI | 0.323263494 | 0.435292819 | 0.382494128 |
| Joint-K+AR | 0.235042128 | 0.440806975 | 0.351642207 |
| FULL-UOI-PSAR | 0.323263494 | 0.435292819 | 0.382494128 |

## Centered-coordinate OOD

- sheet1_to_sheet2/joint_lift: absolute extension `0.050162%` → centered extension `0.023810%`.
- sheet1_to_sheet2/heater_power: absolute extension `4.285714%` → centered extension `0.092117%`.
- sheet1_to_sheet2/crystal_rotation: absolute extension `0.134651%` → centered extension `5.594198%`.
- sheet1_to_sheet2/crucible_rotation: absolute extension `0.000000%` → centered extension `0.000000%`.
- sheet2_to_sheet1/joint_lift: absolute extension `0.285551%` → centered extension `0.082572%`.
- sheet2_to_sheet1/heater_power: absolute extension `7.509994%` → centered extension `0.004045%`.
- sheet2_to_sheet1/crystal_rotation: absolute extension `0.144441%` → centered extension `6.108890%`.
- sheet2_to_sheet1/crucible_rotation: absolute extension `0.534457%` → centered extension `0.009399%`.

## Interpretation boundary

The decimal continuous-EDF coordinate is an effective smoothing complexity, not a physically exact count of degrees of freedom. GCV, REML and L-curve values were retained only as diagnostics and never selected the model. The fitted surfaces remain registered closed-loop input-history response surfaces; two rods do not establish universal open-loop plant causality.
