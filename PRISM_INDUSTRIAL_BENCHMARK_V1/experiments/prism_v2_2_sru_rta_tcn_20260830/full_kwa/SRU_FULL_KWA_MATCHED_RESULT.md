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
- Test RMSE: **NOT_AVAILABLE**
- Test R2 / author-labeled score: **NOT_AVAILABLE**

### corrected_parser
- Test RMSE: **NOT_AVAILABLE**
- Test R2 / author-labeled score: **NOT_AVAILABLE**

## Full PRISM v2.2(beta)

FULL_RESULT_NOT_AVAILABLE — inspect `prism_v22_full_kwa.log`.


## Upstream SHA256

```text
98852f52b4179c000fbb67abdcac15a0d6eb67e19cd4e5d7e0c6639f417a37fc  RTA-TCN-SRU.author.py
82faa3242e58e74b15020bc499ff913bbd3576869586e6f6d911b79d4664a826  SRU_data.txt
```

## Scientific note

K/W/A selection is frozen from author-train development/calibration data. The final RTA author test targets are not used for PRISM route, candidate, penalty, Gamma, W, or A selection.
