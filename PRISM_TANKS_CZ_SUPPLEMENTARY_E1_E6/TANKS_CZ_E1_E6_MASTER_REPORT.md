# Tanks + CZ supplementary E1–E6 master report

## Current status

- Authority base: prism-strict-oof-finalization-20260915 at 2ee6273b8f915cbcdff2f46d56bc80047ddae4a7.
- Cascaded Tanks E1 is completed on the official H16/64-second direct-level task.
- Cascaded Tanks E2–E6 were corrected and rerun under protocol PRISM_V211_CASCADED_TANKS_E2_E6_H16_STRICT_OOF_20260922_R3.
- CZ raw-2s L256 H-scan is complete for h=1/2/4/8/16.
- Corrected CZ raw-2s H4 E1–E6 R2 completed on execution commit 6de39fe; aggregate-only evidence is published under `CZ_RAW2S_H4_CORRECTED_R2/`.
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

Corrected H4 E1–E6 summary:

- E1: strict nested OOF selected only A for both dynamic directions. Formal dynamic RMSE / reconstructed level R2 are 0.015128 / 0.998318 (Rod1 to Rod2) and 0.016130 / 0.999208 (Rod2 to Rod1).
- E2: semisynthetic overall recovery rate 0.6375 and false-admission rate 0.10625.
- E3: mean equal-budget channel-specific multiscale gain 0.2663%; individual seeds include small negative gains.
- E4: the best candidate-universe variants are STANDARD for Rod1 to Rod2 and EXPANDED for Rod2 to Rod1; family ablations were independently refit.
- E5: 118 conditions were evaluated across E1–E4 evidence.
- E6: N1 frozen and N2 reidentified each contain 200 rows; at 10% input noise the N1 dynamic RMSE means are 0.016649 and 0.017668.
