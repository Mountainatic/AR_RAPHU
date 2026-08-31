# PRISM v2.2 SRU full KWA — corrected inherited C contract

> Status: **POST_LOCKBOX_IMPLEMENTATION_CORRECTION_DESCRIPTIVE**. This is not a clean new confirmatory lockbox result.

## Correction

- C ridge restored to numerical-stability-only semantics.
- Smallest stable registered ridge is used.
- Frozen v2.1.1 OOF input-path-preservation gate is restored.
- Collapsed compressed C falls back to BEST_ACTIVE_K instead of killing the entire temporal branch.
- No new thresholds were introduced.

## Branches

- D: eligible=True; reason=C_FALLBACK_TO_BEST_ACTIVE_K; active=[0, 2]; C_family=BEST_ACTIVE_K_CHANNEL; alpha=1e-08; fallback=True
- M: eligible=True; reason=C_FALLBACK_TO_BEST_ACTIVE_K; active=[0, 2]; C_family=BEST_ACTIVE_K_CHANNEL; alpha=1e-08; fallback=True
- S: eligible=True; reason=PASS; active=[2]; C_family=ADDITIVE_COMPRESSED; alpha=1e-08; fallback=False

## Assembly

- Gamma weights: `{'D': 0.02553628157253413, 'M': 0.02575607477958707, 'S': 0.020052448815215174, 'PERSISTENCE': 0.9286551948326637}`
- W selected: `('NATURAL_CUBIC_CORRECTION', 4, 1.0, 30.0, 1)`; active=True
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 4328.7612810830615, 30.0)`; active=True

## Nested test routes

- K_GAMMA: RMSE=0.018562782; MAE=0.005953501; R2=0.896871333
- K_GAMMA_W: RMSE=0.016936661; MAE=0.006074193; R2=0.914148306
- K_GAMMA_W_A: RMSE=0.014768968; MAE=0.005414807; R2=0.934717980
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
- correction_id=PRISM_V2_2_SRU_C_CONTRACT_CORRECTION_20260831_V1
- confirmatory_use=False
