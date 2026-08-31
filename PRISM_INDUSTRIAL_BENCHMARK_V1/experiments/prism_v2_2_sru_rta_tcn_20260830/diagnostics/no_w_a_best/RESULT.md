# SRU v2.2 no-W best-mean A diagnostic

Strict K and best-mean C are retained; W is forced identity; A chooses its minimum-mean OOF candidate only if it passes the existing activation guard.

- Gamma weights: `{'D': 0.023700651548152225, 'M': 0.02363524265062974, 'S': 0.01990973971455019, 'PERSISTENCE': 0.9327543660866678}`
- W selected: `IDENTITY_CORRECTION`; active=False
- A selected: `('MATURE_RESIDUAL_AR', (1, 64), 5.336699231206313, 30.0)`; active=True
- A best candidate: `('MATURE_RESIDUAL_AR', (1, 64), 5.336699231206313, 30.0)`; best OOF mean MSE=7.803016286463115e-05

- K_GAMMA: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W_A: RMSE=0.013149621; MAE=0.005039131; R2=0.948248879
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
