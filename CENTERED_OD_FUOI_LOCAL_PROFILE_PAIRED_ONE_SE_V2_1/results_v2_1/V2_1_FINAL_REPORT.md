# Centered OD-FUOI Local Profile + Paired One-SE V2.1 Final Report

- Selection status: `SELECTION_RESOLVED_V2_1`
- Estimator status: `ESTIMATOR_STABLE_V2_1`
- Model registration: `CENTERED_FULL_URYSOHN_REJECTED`
- Nonlinear increment: `NONLINEAR_INCREMENT_NOT_CERTIFIED`
- Residual: `RESIDUAL_EXACT_ZERO`

## Local minimum and paired one-SE

- sheet1_to_sheet2: d_min=`7.417023`, d_P1SE=`6.851919`, V2 d_1SE=`1.067359`, change=`5.784560`, paired delta=`0.00129698101`, paired SE=`0.0012917819`.
- sheet2_to_sheet1: d_min=`3.191193`, d_P1SE=`2.151581`, V2 d_1SE=`17.744388`, change=`-15.592807`, paired delta=`0.00167460163`, paired SE=`0.00166730371`.

## Frozen L6 outer results

| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |
|---|---:|---:|---:|
| Persistence | 0.314405304 | 0.658043636 | 0.513060810 |
| old K-only | 0.325154539 | 0.486323887 | 0.412382911 |
| NLinear-U | 0.217866830 | 0.472308647 | 0.365854318 |
| R1-LIN-DERIVED | 0.358359552 | 0.640553518 | 0.516801401 |
| LIN-UOI | 0.357565274 | 0.642132711 | 0.517485433 |
| FULL-UOI | 0.409988552 | 0.640546860 | 0.535943170 |
| Joint-K+AR | 0.235042128 | 0.440806975 | 0.351642207 |
| FULL-UOI-PSAR | 0.409988552 | 0.640546860 | 0.535943170 |

## Interpretation

V2.1 did not change the centered Full-Urysohn model. It replaced the whole-domain interpolation gate with deterministic log-excess basin discovery plus local minimum certification, and replaced absolute fold-MSE one-SE with paired 40-minute block one-SE on the same validation samples. Far-field interpolation remains diagnostic and is not a selection gate. Decimal EDF values are effective smoothing coordinates, not physical degrees of freedom.
