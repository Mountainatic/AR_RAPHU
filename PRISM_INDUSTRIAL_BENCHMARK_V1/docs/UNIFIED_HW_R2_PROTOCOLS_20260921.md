# Unified H/W and R² protocol (2026-09-21)

This is the single integration note for the six public dataset families plus
the private CZ raw-2s extension.  The machine-readable authority is
`configs/unified_hw_r2_protocols_20260921.json`; the executable index rules are
in `src/prism_benchmark/unified_hw_protocol.py`.

## Code authority and evidence boundaries

- PRISM method authority: branch `prism-strict-oof-finalization-20260915`,
  commit `2ee6273b8f915cbcdff2f46d56bc80047ddae4a7`.
- Tanks supplementary base: commit
  `5634c6ffc7c71095726b0718aa173b610aa8a368`, which descends from the method
  authority commit.
- CZ zero-C Joint routing correction: audited server commit
  `f49f7845fe6dbe6b8689bc9b01d3dbe35e8360a0`, reproduced in this integration.
- The five original public families below are strict nested-OOF
  development/validation results with `test_accessed=false`.
- Cascaded Tanks is a frozen independent `uVal/yVal` test result and an
  E1-compatible extension, not a native full-registry E1 unit.
- CZ raw data and frozen private artifacts remain private.  Only protocol,
  code and derived aggregate metrics are published here.

## Primary protocols and frozen results

All H/W/W0 values are counts of the dataset's native samples.

| Dataset/head | cadence | H/W/W0 | registered target | evidence | RMSE | delta R² | level R² |
|---|---:|---:|---|---|---:|---:|---:|
| TEP G nowcast | 180 s | 0/1/1 | `D[t]-D[t-1]` | nested OOF | 0.282581 | 0.198090 | 0.771139 |
| Debutanizer C4 | 360 s | 5/1/1 | `y[t+5]-y[t-1]` | nested OOF | 0.026063 | 0.913745 | 0.973804 |
| SRU H2S | 60 s | 1/1/1 | `y[t+1]-y[t-1]` | nested OOF | 0.022546 | 0.370179 | 0.807342 |
| SRU SO2 | 60 s | 1/1/1 | `y[t+1]-y[t-1]` | nested OOF | 0.022112 | 0.153906 | 0.793937 |
| PMSM PM5 | 0.5 s | 600/60/60 | `mean(y[t+600:t+660])-mean(y[t-60:t])` | nested OOF | 2.905076 | 0.517849 | 0.976964 |
| MetroPT P60 | 10 s | 6/1/1 | `y[t+6]-y[t-1]` | nested OOF | 0.187031 | 0.699218 | 0.924440 |
| MetroPT Oil20 | 10 s | 120/12/12 | `mean(y[t+120:t+132])-mean(y[t-12:t])` | nested OOF | 3.176341 | 0.508312 | 0.727743 |
| Cascaded Tanks | 4 s | 16/1/N/A | direct `y[origin+16]` | frozen test | 0.392143 | N/A | 0.965678 |

The Tanks R² is not unexpectedly low and was not misread.  Its value is a
direct-level R² over the validation level variance.  It must not be compared
as if it were delta R².  The selected `K+C+DELTA_W` stage improves test RMSE
from 0.526545 to 0.392143 while level R² rises from 0.938120 to 0.965678.

## Universal delta reporting contract

For all public delta heads except the special CZ convention:

```text
input history = strictly before t
current_level = mean(y[t-W0:t])
future_level  = mean(y[t+H:t+H+W])
delta_target  = future_level - current_level
level_pred    = current_level + delta_pred
```

All slices are left-closed/right-open.  The prediction residual is identical
in delta and reconstructed-level spaces, so MSE, RMSE and MAE must match.
R² does not generally match because delta and level targets have different
variance.  The primary reporting R² is reconstructed level R²; delta R² is
reported as the secondary change-prediction diagnostic.

The historical field `persistence_skill` remains the MSE-relative definition:

```text
persistence_skill_mse  = 1 - MSE_model / MSE_persistence
persistence_skill_rmse = 1 - RMSE_model / RMSE_persistence
```

Both names are emitted explicitly.  The frozen CZ formal report's historical
`persistence_skill` field is MSE-relative; this release binds it to
`persistence_skill_mse` and derives the separately named RMSE-relative value.

## Dataset-specific constraints

- TEP is a nowcast: `[t-L,t)` is the input, `D[t-1]` is the anchor, and the
  current sample `D[t]` never enters the input.  Candidate histories are 128
  and 256 samples (6.4 h and 12.8 h).
- Debutanizer neural history candidates are 20/40/80 samples (2/4/8 h).
- SRU H2S and SO2 are independent heads; history candidates are 120/240/480
  samples (2/4/8 h) and choices cannot be shared across heads.
- PMSM splits by complete `profile_id`; a profile cannot cross a split.
- MetroPT P60 uses `proxy_excluded` for the primary result.  `full_sensor` is
  secondary sensitivity only.
- Tanks observes samples through `origin` and predicts the direct level at
  `origin+16`; its W0 is therefore not a delta-anchor window.

## Private CZ raw-2s protocol

```text
sampling interval = 2 seconds
history           = 256 samples
input             = [t-256,t)
anchor            = D[t-1]
target_delta      = D[t+h-1] - D[t-1]
level_prediction  = D[t-1] + delta_pred
W/W0              = 1/1
```

Rod1→Rod2 and Rod2→Rod1 are trained and selected independently.  The primary
task is h=4 (8 seconds); the complete registered scan is:

| h | real horizon | Rod1→Rod2 skill (MSE / RMSE) | Rod2→Rod1 skill (MSE / RMSE) |
|---:|---:|---:|---:|
| 1 | 2 s | 0.11709 / 0.06037 | 0.14908 / 0.07755 |
| 2 | 4 s | 0.17364 / 0.09096 | 0.18846 / 0.09914 |
| 4 | 8 s | 0.26679 / 0.14372 | 0.26538 / 0.14290 |
| 8 | 16 s | 0.31629 / 0.17313 | 0.33713 / 0.18583 |
| 16 | 32 s | 0.35710 / 0.19819 | 0.39132 / 0.21982 |

At primary h=4, Joint gives:

| direction | RMSE | reconstructed level R² | skill MSE | skill RMSE |
|---|---:|---:|---:|---:|
| Rod1→Rod2 | 0.013712 | 0.998618 | 0.266791 | 0.143724 |
| Rod2→Rod1 | 0.014822 | 0.999331 | 0.265383 | 0.142902 |

These are private-data aggregate results and do not make the raw workbook or
sample-level predictions public.

CZ materialization, segment/purge logic and the exact
`D[t+h-1]-D[t-1]` target live in `src/prism_benchmark/cz_l256_nowcast.py`.
The corrected private H4 entry point `scripts/run_cz_raw2s_e1_e6.py` now
delegates K/C/W/A/Joint fitting and checkpoint replay to
`scripts/run_independent_extension_20260825.py` and verifies authority-module
Git blobs before data access.  The earlier custom Ridge/PCA E1–E6 output is
invalid for authoritative-PRISM claims; see
`docs/CZ_RAW2S_AUTHORITY_CORRECTION_20260922.md`.  The runner requires a
private `raw_root` at execution time and never embeds or uploads the workbook.

## E1–E6 interpretation

E1 is the stagewise ablation (`K`, `K+C`, `K+C+ΔW`,
`K+C+ΔW+A`/Joint).  E2–E6 are supporting mechanism, multiscale, sensitivity,
structural-stability and measurement-robustness experiments.  They do not
define H/W and must inherit the frozen task head.  Therefore results from an
old CZ D20 head cannot be mixed with raw-2s h=1/2/4/8/16 results.

The zero-C correction in `v211_joint_stability.py` is important for E1: an
admissible zero C increment is the additive identity, so Joint must use the
routed seed instead of indexing a nonexistent `best_active_k_channel`.

For Tanks, the runnable files are
`scripts/run_cascaded_tanks_w_experiment.py`,
`scripts/build_cascaded_tanks_e1_stagewise.py` and
`scripts/tanks_validated_e2_e6.py`.  The common PRISM E1–E6 implementation is
under `src/prism_benchmark/`; the registry's `code_map` gives the authoritative
entry points.

## Reproduction and audit commands

Run the protocol and metric tests:

```bash
python -m pytest tests/test_level_reconstruction.py tests/test_unified_hw_protocol.py
```

Report a delta head from a CSV containing
`delta_true,delta_pred,current_level`.  To verify the H/W target rather than
only calculate metrics, also include `origin` and pass a one-column source CSV
whose column is named `value`:

```bash
PYTHONPATH=src python scripts/report_unified_hw_r2.py \
  --head-id METRO_P60__H6__W1 \
  --predictions /path/to/predictions.csv \
  --series /path/to/source_series.csv \
  --output /path/to/metrics.json
```

For Tanks/direct-level heads, the input CSV columns are `y_true,y_pred`; target
verification additionally requires `origin` and `--series`.

## Historical protocols excluded from the primary ranking

- TEP H1/W2/W0=2 and H4/W2/W0=2.
- SRU H5/W1/W0=1.
- CZ D20 at 10-second modeling scale, H120/W12/W0=12.

They remain compatibility evidence only and must not be ranked together with
the primary table.
