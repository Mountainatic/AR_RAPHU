# SRU v2.2 KGamma-A no-W diagnostic

Best-mean C alpha is retained; W is forced to IDENTITY to isolate whether K/Gamma itself helps or harms the mature residual A route.

- Gamma weights: `{'D': 0.023700651548151593, 'M': 0.023635242650628965, 'S': 0.019909739714552954, 'PERSISTENCE': 0.9327543660866665}`
- W selected: `IDENTITY_CORRECTION`; active=False
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 151.99110829529332, 30.0)`; active=True

- K_GAMMA: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W_A: RMSE=0.013397758; MAE=0.004523108; R2=0.946277333
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
