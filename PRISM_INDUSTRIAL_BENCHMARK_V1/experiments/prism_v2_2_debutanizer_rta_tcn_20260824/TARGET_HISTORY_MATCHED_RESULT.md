# PRISM v2.2(beta) vs RTA-TCN — target-history-enabled Debutanizer

Status: benchmark-branch evidence only.

## Information contract
Both models receive the same 40-sample window of 7 process variables plus one strictly causal previous-target channel `x8[t]=y[t-1]`. The current target `y[t]` is never an input. Parser is corrected with `header=None`; train boundary and RTA-TCN training settings are otherwise unchanged.

## PRISM results
- Persistence-only: RMSE `0.014150054`, R2 `0.993874465`.
- D/M/S only: RMSE `0.009480649`, R2 `0.997250182`.
- D/M/S + legal persistence anchor: RMSE `0.008362955`, R2 `0.997860327`.
- D/M/S weights: `{'D': 0.4715945153401385, 'M': 0.06276646275138512, 'S': 0.46563902190847634}`.
- D/M/S/P weights: `{'D': 0.3436908881839403, 'M': 0.04910970652105538, 'S': 0.34044070712948865, 'P': 0.26675869816551573}`.
- Selected lambdas: `{'D': 0.0001, 'M': 1.0, 'S': 0.0001}`.
- Numerical audits: `{'D': {'features': 56, 'rank': 56, 'condition': 357.63751766800283}, 'M': {'features': 48, 'rank': 48, 'condition': 518.9548061364669}, 'S': {'features': 48, 'rank': 48, 'condition': 9046.77056209604}}`.

## RTA-TCN result with same previous-target channel
- Train RMSE: `0.003932258787847058`
- Train R2: `0.9993112333783409`
- Test RMSE: `0.006580559708018379`
- Test R2: `0.998675190267718`

## Important interpretation
This is a legal past-target-state experiment, not target leakage: `y[t]` is never supplied to either model. The persistence anchor uses only `y[t-1]`.
