# SRU v2.2 best-mean A diagnostic

Strict K and best-mean C are retained; W uses the current guarded selector; A chooses its minimum-mean OOF candidate only if that candidate passes the existing activation guard.

- Gamma weights: `{'D': 0.023700651548152225, 'M': 0.02363524265062974, 'S': 0.01990973971455019, 'PERSISTENCE': 0.9327543660866678}`
- W selected: `('MONOTONE_I_SPLINE_CORRECTION', 4, 1.0, 30.0, 1)`; active=True
- A selected: `('MATURE_RESIDUAL_AR', (1, 16), 0.18738174228603832, 30.0)`; active=True

- A best candidate: `('MATURE_RESIDUAL_AR', (1, 16), 0.18738174228603832, 30.0)`; best OOF mean MSE=7.320907815011237e-05

- K_GAMMA: RMSE=0.018564896; MAE=0.005961232; R2=0.896847843
- K_GAMMA_W: RMSE=0.016619207; MAE=0.005802037; R2=0.917336484
- K_GAMMA_W_A: RMSE=0.012424551; MAE=0.004820441; R2=0.953798646
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
