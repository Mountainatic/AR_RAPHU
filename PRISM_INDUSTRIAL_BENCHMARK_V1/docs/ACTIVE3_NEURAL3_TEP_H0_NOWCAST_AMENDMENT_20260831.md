# Active-3 Neural-3 TEP H0 nowcast amendment

Status: `FROZEN_AFTER_TEP_H0_NOWCAST_REPLACEMENT_BEFORE_GPU_DEVELOPMENT`

This amendment replaces the active TEP three-minute forecast head in the GPU
matrix with a strict-past estimate of the current TEP state. It does not alter
the SRU or CZ forecast heads. Consequently, the resulting collection is a
heterogeneous task matrix: TEP is a current-value nowcast, while the other
datasets remain future forecasts. It must not be described as a homogeneous
seven-forecast-head leaderboard.

## Active TEP protocol

```text
task/head       = TEP_G_NOWCAST_H0__H0__W1
cadence         = 180 seconds
history         = {128, 256} points
input           = [t-L,t)
current t       = strictly excluded from input
anchor          = D[t-1]
target_delta    = D[t]-D[t-1]
level_pred      = D[t-1]+delta_pred
horizon         = 0 minutes
proxy policy    = proxy_excluded
common support  = L256 target rows within each view
```

`L128` covers 6.4 hours and `L256` covers 12.8 hours. Both histories remain in
the GPU development grid. The CPU development selection counts (34 for L128 and
23 for L256) are not sufficient to freeze every GPU model to L128.

The active views are:

1. `input_only / record_time`: no target history enters the neural input.
2. `dynamic / record_time`: strictly past, record-time target history is allowed.
3. `dynamic / analyzer_maturity_5_steps`: target history is delayed by five
   samples (15 minutes).

If `D[t-1]` or recent target history is not available at deployment time,
`dynamic / record_time` is an ideal-availability upper bound, not the deployment
headline. The deployment result must then use input-only or maturity-5.

## Frozen reference results and grading thresholds

| View | PRISM Level R2 | PRISM Delta R2 | Persistence skill |
|---|---:|---:|---:|
| input-only / record-time | 0.713246 | approximately 0 | approximately -0.000001 |
| dynamic / maturity-5 | 0.713496 | approximately 0 | approximately -0.0000005 |
| dynamic / record-time | **0.781337** | **0.237455** | **0.237455** |

The best existing CPU result under the same protocol is ARX with Level R2
`0.836777`, Delta R2 `0.430792`, and persistence skill `0.430792`.

The GPU report therefore evaluates these explicit thresholds:

- Level R2 above `0.781337`: exceeds PRISM.
- Level R2 above `0.836777`: exceeds every existing CPU model.
- Persistence skill above `0.237455`: exceeds PRISM's change prediction.
- Persistence skill above `0.430792`: exceeds the best existing ARX.

Level R2 above 0.78 is not sufficient by itself because persistence already has
Level R2 of approximately 0.713.

## Compatibility appendix

The prior three-minute TEP task remains registered for continuity but is not an
active GPU head:

```text
TEP_G_REP_H1__H1__W2
best PRISM dynamic Level R2 = 0.523519
```

## Evidence

- [TEP H0 nowcast aggregate results](https://github.com/Mountainatic/AR_RAPHU/releases/download/prism-v2-1-1-tep-sru-cz-l256-formal-20260825/TEP_CPU_PRISM_NOWCAST_H0_W1_HISTORY_128_256_20260828_AGGREGATE_ONLY.tar.gz)
- [Protocol and reproduction notes](https://github.com/Mountainatic/AR_RAPHU/releases/download/prism-v2-1-1-tep-sru-cz-l256-formal-20260825/TEP_CPU_PRISM_NOWCAST_H0_W1_HISTORY_128_256_20260828_README.md)

