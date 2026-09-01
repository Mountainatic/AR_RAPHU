# PRISM v2.2(beta) full KWA vs RTA-TCN — SRU

## Scope

- RTA-TCN is rerun from the official author code on the official SRU file.
- The exact-author parser and corrected `header=None` parser are reported separately.
- The new PRISM result is the full beta chain: `D/M/S -> K/C -> Gamma_CT -> DeltaW -> A`.
- The previous D/M/S + Ridge experiment is a temporal-representation ablation and is not labeled as full PRISM here.
- SRU cadence is 60 s; CT tau is frozen in physical time as [60,120,240,480,960,1440] s.

## Information-set warning

- Official RTA-TCN reference: input-only; historical target y is forbidden.
- Full PRISM KWA: dynamic record-time information; only y[t-1] or older and mature residuals are legal.
- Therefore RTA-TCN vs full KWA is not claimed to be strictly information-matched. It is a model/task reference plus a full-architecture result.

## RTA-TCN official-code rerun

### exact_author
- Test RMSE: **0.025544488**
- Test R2 / author-labeled score: **0.804771079**

### corrected_parser
- Test RMSE: **0.030893714**
- Test R2 / author-labeled score: **0.714350461**

## Full PRISM v2.2(beta)

- **K_GAMMA**: RMSE 0.018864074, MAE 0.006103631, R2 0.893496409
- **K_GAMMA_W**: RMSE 0.018864074, MAE 0.006103631, R2 0.893496409
- **K_GAMMA_W_A**: RMSE 0.013539937, MAE 0.004564453, R2 0.945131061
- **PERSISTENCE**: RMSE 0.018864074, MAE 0.006103631, R2 0.893496409

- Gamma weights: `{'__PERSISTENCE_ONLY_ZERO_DELTA__': 4.547473508864641e-13, 'PERSISTENCE': 0.9999999999995453}`
- W selected: `IDENTITY_CORRECTION`; active=False
- A selected: `('MATURE_RESIDUAL_AR', (1, 4), 151.99110829529332, 30.0)`; active=True
- Test target used for selection: `False`
- Runtime: 155.130 s

### Branch admission

- D: eligible=False; active K channels=[0, 2]; reason=C_INPUT_PATH_NOT_PRESERVED
- M: eligible=False; active K channels=[0, 2]; reason=C_INPUT_PATH_NOT_PRESERVED
- S: eligible=False; active K channels=[2]; reason=C_INPUT_PATH_NOT_PRESERVED

## Upstream SHA256

```text
98852f52b4179c000fbb67abdcac15a0d6eb67e19cd4e5d7e0c6639f417a37fc  RTA-TCN-SRU.author.py
82faa3242e58e74b15020bc499ff913bbd3576869586e6f6d911b79d4664a826  SRU_data.txt
```

## Scientific note

K/W/A selection is frozen from author-train development/calibration data. The final RTA author test targets are not used for PRISM route, candidate, penalty, Gamma, W, or A selection.
