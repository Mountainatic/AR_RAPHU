# PRISM v2.2(beta) vs RTA-TCN — SRU matched benchmark

## Frozen information contract

- Upstream source: `CHM00/RTA-TCN-and-JITL-RTA-TCN`.
- Data: official `SRU/SRU_data.txt`.
- Author task: H2S current-time soft sensing.
- Inputs: 5 process variables.
- Sequence length: 25.
- Historical target y: forbidden.
- Future inputs: forbidden.
- Author train target count: 7039.
- RTA-TCN: author architecture/training code, 240 epochs, batch 64, lr 0.001, seed 1024.
- PRISM branch profile: D lags [0,1,2,4,8,16,24], CT taus [1,2,4,8,16,24].

## RTA-TCN official-code rerun

### exact_author
- Train RMSE: 0.012457266
- Train R2: 0.939762614
- Test RMSE: **0.025544488**
- Test R2: **0.804771079**

### corrected
- Train RMSE: 0.011232580
- Train R2: 0.951024884
- Test RMSE: **0.030893714**
- Test R2: **0.714350461**

## PRISM v2.2(beta)

### exact_author
- D: RMSE 0.036130951, R2 0.609421325
- M: RMSE 0.038910084, R2 0.547025161
- S: RMSE 0.036057834, R2 0.611000526
- Gamma D/M/S: RMSE 0.036511087, R2 0.601159494
- Full-25-lag Ridge: RMSE 0.036211377, R2 0.607680580
- Gamma weights: {'D': 0.32698349356072315, 'M': 0.3325343765228368, 'S': 0.34048212991643995}
- Numerical audits: {'D': {'features': 35, 'rank': 35, 'condition': 235.2381489745978}, 'M': {'features': 30, 'rank': 30, 'condition': 607.994696178835}, 'S': {'features': 30, 'rank': 30, 'condition': 17281.82602398456}}

### corrected
- D: RMSE 0.036126251, R2 0.609393840
- M: RMSE 0.038898570, R2 0.547143556
- S: RMSE 0.036048132, R2 0.611081303
- Gamma D/M/S: RMSE 0.036503167, R2 0.601200708
- Full-25-lag Ridge: RMSE 0.036205055, R2 0.607687882
- Gamma weights: {'D': 0.326922144258643, 'M': 0.3325414797438401, 'S': 0.34053637599751696}
- Numerical audits: {'D': {'features': 35, 'rank': 35, 'condition': 235.22145360471856}, 'M': {'features': 30, 'rank': 30, 'condition': 607.6982648022944}, 'S': {'features': 30, 'rank': 30, 'condition': 17231.62710637447}}

## Secondary PRISM-only SO2 audit

SO2 is evaluated with the identical frozen PRISM protocol as a secondary target; it is not presented as an official RTA-TCN matched task unless the author script is separately verified for SO2.

- exact_author: Gamma RMSE 0.025317029, R2 0.786107708; best branch R2 0.794935816.
- corrected: Gamma RMSE 0.025317254, R2 0.786078870; best branch R2 0.794906910.

## Upstream SHA256

```text
98852f52b4179c000fbb67abdcac15a0d6eb67e19cd4e5d7e0c6639f417a37fc  RTA-TCN-SRU.author.py
82faa3242e58e74b15020bc499ff913bbd3576869586e6f6d911b79d4664a826  SRU_data.txt
```

## Scientific note

Exact-author and corrected-parser results are reported separately. No test-set result is used for model or hyperparameter selection.
