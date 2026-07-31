# OD-FUOI NLinear Projection CPU Confirm V1 Final Report

- Registration: `FULL_URYSOHN_REJECTED_ON_CURRENT_DATA`
- NLinear projection: `NLINEAR_PREDICTIVE_ONLY`
- Nonlinear increment: `NONLINEAR_INCREMENT_NOT_CERTIFIED`
- Residual: `RESIDUAL_EXACT_ZERO`

## Frozen L6 results

| Model | Sheet1→Sheet2 RMSE | Sheet2→Sheet1 RMSE | Pooled RMSE |
|---|---:|---:|---:|
| Persistence | 0.314405304 | 0.658043636 | 0.513060810 |
| old K-only | 0.325154539 | 0.486323887 | 0.412382911 |
| Dynamic-PLS | 0.324895548 | 0.464359613 | 0.399628067 |
| NLinear-U | 0.217866830 | 0.472308647 | 0.365854318 |
| R1-LIN-DERIVED | 11.914187238 | 3.738942249 | 8.888166573 |
| LIN-UOI | 14.855853310 | 14.875244293 | 14.865395018 |
| FULL-UOI | 29.259986360 | 246.094879086 | 173.856760904 |
| Temporal Autoencoder | 0.221369529 | 0.479772554 | 0.371654003 |
| Joint-K+AR | 0.235042128 | 0.440806975 | 0.351642207 |
| FULL-UOI-PSAR | 29.259986360 | 246.094879086 | 173.856760904 |

## Derived structure

- GCV lambda: `{"sheet1_to_sheet2": 1.0000007648900382e-12, "sheet2_to_sheet1": 1.0000007648900382e-12}`
- Effective df: `{"sheet1_to_sheet2": 719.3611717340987, "sheet2_to_sheet1": 693.8792188148295}`
- Rank-1 energy ratios: `{"sheet1_to_sheet2": 0.8275288917584288, "sheet2_to_sheet1": 0.9130031161585737}`
- Channel surface states: `{"crucible_rotation": "UNRESOLVED", "crystal_rotation": "PREDICTIVE_ONLY", "heater_power": "UNRESOLVED", "joint_lift": "UNRESOLVED"}`
- Residual models: `{"sheet1_to_sheet2": "A0", "sheet2_to_sheet1": "A0"}`

## Paired 40-minute block bootstrap

- Full vs old K: median relative MSE improvement `-17626841.800734%`, positive probability `0.000000`.
- Full vs NLinear: median relative MSE improvement `-22930304.439076%`, positive probability `0.000000`.
- Full+PSAR vs Joint-K+AR: median relative MSE improvement `-23322078.409402%`, positive probability `0.000000`.

## Scientific boundary

The fitted objects are registered closed-loop input-history response surfaces. Pooled RMSE, Rank-1 energy, or a visually smooth surface alone does not prove an open-loop plant mechanism or universal cross-rod causality.
