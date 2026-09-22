# Tanks + CZ supplementary E1–E6 master report

## Current status

- Authority base: prism-strict-oof-finalization-20260915 at 2ee6273b8f915cbcdff2f46d56bc80047ddae4a7.
- Cascaded Tanks E1 is completed on the official H16/64-second direct-level task.
- Cascaded Tanks E2–E6 were corrected and rerun under protocol PRISM_V211_CASCADED_TANKS_E2_E6_H16_STRICT_OOF_20260922_R3.
- CZ raw-2s L256 H-scan is complete for h=1/2/4/8/16.
- Corrected CZ raw-2s H4 E1–E6 code is committed, but its new private-data R2 run remains NOT_YET_RUN.
- Historical Tanks E2–E6 R2 and CZ D20 artifacts are retained only for audit and excluded from current claims.

## Cascaded Tanks

E1 frozen validation:

| stage | RMSE | level R2 |
|---|---:|---:|
| K | 0.526545 | 0.938120 |
| K+C (C identity) | 0.526545 | 0.938120 |
| K+C+DELTA_W | 0.392143 | 0.965678 |
| K+C+DELTA_W+A (A identity) | 0.392143 | 0.965678 |

Corrected E2–E6:

- E2 stage-vector accuracy 0.4467; NULL false-admission 0.30.
- E3 equal-budget stage-specific multiscale gain 7.0353%; both arms use the same K/C/W/A family and candidate count.
- E4A/E4B now refit every candidate/family condition on the same H16 support.
- E5 selected one stable stage vector across ten perturbation seeds.
- E6 N1/N2 was rerun across five perturbation types and four magnitudes; a 0.10 stage-flip probability occurred only for 10% random-walk perturbation.

## Private CZ

The current primary protocol is 2-second sampling, L256, H4/W1/W0=1, with target D[t+3]-D[t-1]. At H4 the frozen H-scan gives:

| direction | RMSE | reconstructed level R2 | skill MSE | skill RMSE |
|---|---:|---:|---:|---:|
| Rod1 to Rod2 | 0.013712 | 0.998618 | 0.266791 | 0.143724 |
| Rod2 to Rod1 | 0.014822 | 0.999331 | 0.265383 | 0.142902 |

The previous 0.266791/0.265383 values are MSE-relative skills, not RMSE-relative skills. The old packaged D20 E1–E6 results do not answer the current raw-2s H4 task.
