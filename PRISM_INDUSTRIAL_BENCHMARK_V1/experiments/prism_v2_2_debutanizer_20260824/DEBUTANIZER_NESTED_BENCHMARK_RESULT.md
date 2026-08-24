# PRISM v2.2(beta) — Debutanizer nested multi-step benchmark

## Status

This file belongs only to the benchmark branch `prism-v2-2-beta-debutanizer-benchmark-20260824`.
It does **not** modify or redefine the frozen `prism-v2-2-beta-ct` branch.

The purpose is a small public-dataset check against published multi-step soft-sensor methods.

## Dataset

Fortuna Debutanizer column public benchmark:

- 2394 samples;
- 7 easy-to-measure process inputs plus C4 target;
- public target series is already translated by 8 samples to compensate analyzer delay;
- 2024 MSA-HDMDc paper reports a 6 min cadence and four operating months: March, May, July, September.

For this near-replication the 2394-point public sequence is divided chronologically into four quarters representing the four months. CT state and delay history are reset at every quarter boundary; no state is carried across month boundaries.

## Information set

PRISM v2.2(beta) uses only information available at the forecast origin:

- current and historical process inputs;
- causally available current/historical target state;
- **no future process-input trajectory**.

This is stricter than the published MSA-HDMDc rollout, whose reconstruction equations use exogenous `u[j+k]` at future rollout steps. Therefore the numerical comparison is highly relevant but not a perfectly identical information-set comparison.

## Frozen benchmark instantiation

The silicon-specific second-valued tau bank is not reused. Instead this benchmark uses a fixed, preregistered, sampling-relative dyadic bank:

`tau / dt = [1, 2, 4, 8, 16, 32]`

which at `dt = 6 min` corresponds to:

`tau = [6, 12, 24, 48, 96, 192] min`.

No tau is learned from test data.

Temporal branches:

- `D`: explicit discrete-delay bank at the same dyadic scales;
- `M`: CT multiresolution adjacent-scale increments;
- `S`: CT absolute-state bank;
- `P`: persistence fallback in constrained `Gamma_CT` assembly.

All three learned temporal branches pass the provisional standardized condition-number limit `1e8` on this dataset. Typical condition numbers are about 487–490 (D), 329–338 (M), and 4633–4769 (S).

## Nested validation protocol

- March + May: branch parameter fitting.
- First half of July: choose branch Ridge lambda from `[1e-4, 1e-3, 1e-2, 1e-1, 1]`.
- Second half of July: fit nonnegative simplex `Gamma_CT` weights.
- September: untouched final test.

Thus the final September target is not used for temporal scale choice, Ridge choice, or assembly weights.

## Published comparison target

Patanè, Sapuppo & Xibilia, *Soft Sensors for Industrial Processes Using Multi-Step-Ahead Hankel Dynamic Mode Decomposition with Control*, Electronics 2024.

Published Debutanizer test R2:

| steps | minutes | ARX | MSA-HDMDc |
|---:|---:|---:|---:|
| 2 | 12 | 0.983 | 0.998 |
| 5 | 30 | 0.854 | 0.974 |
| 10 | 60 | 0.347 | 0.890 |
| 20 | 120 | -1.20 | 0.661 |

## PRISM v2.2(beta) nested result

| horizon | PRISM v2.2 R2 | RMSE | Persistence R2 | MSA-HDMDc R2 | R2 gap vs MSA-HDMDc |
|---:|---:|---:|---:|---:|---:|
| 12 min | **0.997103** | 0.010445 | 0.977588 | **0.998** | -0.000897 |
| 30 min | **0.982813** | 0.025430 | 0.867639 | 0.974 | **+0.008813** |
| 60 min | **0.908146** | 0.058949 | 0.548164 | 0.890 | **+0.018146** |
| 120 min | 0.292107 | 0.165018 | -0.142071 | **0.661** | -0.368893 |

Interpretation:

- 12 min: essentially tied with MSA-HDMDc;
- 30 min: PRISM is higher in R2;
- 60 min: PRISM is higher in R2;
- 120 min: PRISM is substantially worse.

This is **not** a claim of universal or latest SOTA. It is evidence that the beta architecture is competitive with, and at 30–60 min numerically exceeds, a strong published multi-step industrial soft-sensor baseline on this small public dataset.

## Gamma_CT weights

### 12 min

`D=0.49084, M=0.04853, S=0.21136, P=0.24927`

### 30 min

`D=0.68449, M=0.15434, S=0.09584, P=0.06533`

### 60 min

`D=0.49929, M=0.23496, S=0.26575, P=0`

### 120 min

`D=0, M=1, S=0, P=0`

The 120 min collapse to the M branch is a useful failure signal: the present fixed state bank/direct-horizon map is not sufficient for very long-horizon Debutanizer prediction.

## Control experiment: current-time soft sensing

A separate legacy 1556/838 current-time soft-sensing experiment was run using only the seven easy-to-measure process inputs (no historical y). PRISM v2.2(beta) obtained approximately:

- R2 = 0.4773;
- RMSE = 0.1394;
- MAE = 0.1065.

This is far below recent Debutanizer current-time soft-sensor results (e.g. 2026 DMCGN reports R2 about 0.9984). Therefore the present evidence does **not** support calling PRISM a general Debutanizer soft-sensing SOTA. Its current strength is specifically multi-step dynamic forecasting.

## Current literature caveat

A newer 2025 Control Engineering Practice method, TiMF (TimeGPT-based Multi-step-ahead Forecasting), is also evaluated on Debutanizer and reports outperforming existing methods. Its accessible abstract does not expose the full Debutanizer numerical table in the sources checked here. Therefore a definitive `latest SOTA` claim is withheld until an exact TiMF/ADRNN numerical head-to-head is reconstructed.

## Bottom line

The scientifically defensible current statement is:

> On a small public Debutanizer multi-step benchmark, a nested-validation PRISM v2.2(beta) near-replication achieves R2 = 0.997/0.983/0.908/0.292 at 12/30/60/120 min. It essentially ties MSA-HDMDc at 12 min and exceeds its published R2 at 30 and 60 min, while losing clearly at 120 min. The comparison is conservative with respect to future exogenous inputs because PRISM does not use them, but the month-boundary split is a near-replication rather than an exact sample-index reconstruction.
