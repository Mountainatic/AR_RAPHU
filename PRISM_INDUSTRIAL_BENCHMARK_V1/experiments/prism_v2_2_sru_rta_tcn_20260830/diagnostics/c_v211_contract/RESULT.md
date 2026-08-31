# SRU v2.2 with v2.1.1-compatible C contract

Diagnostic only. C ridge is numerical-stability-only, smallest stable ridge is selected, OOF input-path preservation is required, and BEST_ACTIVE_K is the terminal C fallback. Strict K, Gamma_CT, W and A are unchanged.

## Branches

- D: eligible=True; reason=C_FALLBACK_TO_BEST_ACTIVE_K; active=[0, 2]; C_family=BEST_ACTIVE_K_CHANNEL; C_alpha=1e-08; fallback=True; path_pass=False; variance_ratio=0.4987558649813998; bestK_variance_ratio=0.5254625805426112; candidate_mse=0.00019739965187363863; bestK_mse=0.00018817371868843223
- M: eligible=True; reason=C_FALLBACK_TO_BEST_ACTIVE_K; active=[0, 2]; C_family=BEST_ACTIVE_K_CHANNEL; C_alpha=1e-08; fallback=True; path_pass=False; variance_ratio=0.4812035332109659; bestK_variance_ratio=0.5187698391974563; candidate_mse=0.00019900956360836039; bestK_mse=0.00018963861876924017
- S: eligible=True; reason=PASS; active=[2]; C_family=ADDITIVE_COMPRESSED; C_alpha=1e-08; fallback=False; path_pass=True; variance_ratio=0.39479587592610554; bestK_variance_ratio=0.35224194968890965; candidate_mse=0.00020377541239979723; bestK_mse=0.00020109474775061265

## Assembly

- Gamma weights: `{'D': 0.02553628157253413, 'M': 0.02575607477958707, 'S': 0.020052448815215174, 'PERSISTENCE': 0.9286551948326637}`
- W selected: `('NATURAL_CUBIC_CORRECTION', 4, 1.0, 30.0, 1)`; active=True
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 4328.7612810830615, 30.0)`; active=True

## Test routes

- K_GAMMA: RMSE=0.018562782; MAE=0.005953501; R2=0.896871333
- K_GAMMA_W: RMSE=0.016936661; MAE=0.006074193; R2=0.914148306
- K_GAMMA_W_A: RMSE=0.014768968; MAE=0.005414807; R2=0.934717980
- PERSISTENCE: RMSE=0.018864074; MAE=0.006103631; R2=0.893496409

- test_target_used_for_selection=False
