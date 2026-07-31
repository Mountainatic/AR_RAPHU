# OPS-UOI Shared–Private K CPU Confirm V1 Final Report

- Registration: `REJECTED`
- Success level: `LEVEL_D_REJECTED`
- Mother space: `V0`
- Shared rank: `0`
- Private channel: `None`
- Nonlinear channel: `None`
- Residual models: `{"sheet1_to_sheet2": "A0", "sheet2_to_sheet1": "AR:2"}`

## Frozen L6 result

The input-only shared/private K has pooled RMSE `0.513060810` and pooled MSE `0.263231394`. The final K→Residual model has pooled RMSE `0.490386836`.

| Direction | Input K RMSE | K→Residual RMSE |
|---|---:|---:|
| sheet1_to_sheet2 | 0.314405304 | 0.314405304 |
| sheet2_to_sheet1 | 0.658043636 | 0.621907086 |

## Certification

- Shared: `PREDICTIVE_SHARED_LOW_RANK_NOT_CERTIFIED`
- Private: `PRIVATE_EXACT_ZERO`
- Nonlinear exact zero: `True`
- Relative MSE improvement versus old K-only: `-54.787671%`
- 40-minute paired-bootstrap positive probability versus old K-only: `0.006000`
- Relative MSE improvement versus NLinear-U: `-96.662391%`
- Relative MSE improvement of final dynamic model versus Joint-K+AR: `-94.480299%`

The physical registration follows the frozen bidirectional and subspace gates; pooled RMSE alone is not used to claim cross-rod physical stability.
