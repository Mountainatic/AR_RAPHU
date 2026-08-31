# SRU v2.2 C-restoration audit

This run restores the already-frozen v2.1.1 SRU C input-path contract while retaining the v2.2 D/M/S, Gamma_CT, W and A candidate universes.

## Nested routes

- K_GAMMA: RMSE=0.018562782, MAE=0.005953501, R2=0.896871333
- K_GAMMA_W: RMSE=0.016936661, MAE=0.006074193, R2=0.914148306
- K_GAMMA_W_A: RMSE=0.014768968, MAE=0.005414807, R2=0.934717980
- PERSISTENCE: RMSE=0.018864074, MAE=0.006103631, R2=0.893496409

- Gamma weights: `{'D': 0.02553628157253413, 'M': 0.02575607477958707, 'S': 0.020052448815215174, 'PERSISTENCE': 0.9286551948326637}`
- W selected: `('NATURAL_CUBIC_CORRECTION', 4, 1.0, 30.0, 1)`; active=True
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 4328.7612810830615, 30.0)`; active=True

## Branch C audit

- D: eligible=True; active K=[0, 2]; reason=C_FALLBACK_TO_BEST_ACTIVE_K_CHANNEL; C family=BEST_ACTIVE_K_CHANNEL; alpha=0.0; fallback=True; gate=INPUT_PATH_COLLAPSED
- M: eligible=True; active K=[0, 2]; reason=C_FALLBACK_TO_BEST_ACTIVE_K_CHANNEL; C family=BEST_ACTIVE_K_CHANNEL; alpha=0.0; fallback=True; gate=INPUT_PATH_COLLAPSED
- S: eligible=True; active K=[2]; reason=PASS; C family=ADDITIVE_COMPRESSED; alpha=1e-08; fallback=False; gate=INPUT_PATH_PRESERVED

## Interpretation rule

Only after C preserves an active K path is W activation/non-activation scientifically interpretable. This audit does not change W/A thresholds based on test results.
