# SRU v2.2 C best-alpha diagnostic

Diagnostic only: C ridge selection uses minimum mean OOF validation loss instead of the largest-alpha one-SE member. All other strict-runner hooks remain unchanged.

## Branches

- D: eligible=True; reason=PASS; active=[0, 2]; C_alpha=151.99110829529332; variance_ratio=0.3222226734394589; variance_pass=True; coefficient_pass=True
- M: eligible=True; reason=PASS; active=[0, 2]; C_alpha=151.99110829529332; variance_ratio=0.3142201977370866; variance_pass=True; coefficient_pass=True
- S: eligible=True; reason=PASS; active=[2]; C_alpha=151.99110829529332; variance_ratio=0.24661740287202544; variance_pass=True; coefficient_pass=True

## Assembly

- Gamma weights: `{'D': 0.023700651548151593, 'M': 0.023635242650628965, 'S': 0.019909739714552954, 'PERSISTENCE': 0.9327543660866665}`
- W selected: `('MONOTONE_I_SPLINE_CORRECTION', 4, 1.0, 30.0, 1)`; active=True
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 4328.7612810830615, 30.0)`; active=True

## Test routes

- K_GAMMA: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W: RMSE=0.016619207; MAE=0.005802037; R2=0.917336484
- K_GAMMA_W_A: RMSE=0.014318031; MAE=0.005106639; R2=0.938643597
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
