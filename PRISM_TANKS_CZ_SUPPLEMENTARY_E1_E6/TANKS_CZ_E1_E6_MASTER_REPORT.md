# Tanks + CZ supplementary E1-E6 master report

## Execution status

- Authoritative branch: prism-strict-oof-finalization-20260915
- Supplementary branch: prism-tanks-cz-supplementary-e1-e6-20260916
- Base commit: 2ee6273b8f915cbcdff2f46d56bc80047ddae4a7
- Cascaded Tanks official archive verified; E1 compatible historical extension completed; E2-E6 validated extension completed under R2.
- CZ E1-E6 frozen R5 evidence reused and labelled supplementary.

## Cascaded Tanks

- E1 K RMSE 0.52654468; K+C RMSE 0.52654468; K+C+W RMSE 0.39214275; K+C+W+A RMSE 0.39214275.
- Historical-compatible W gain: 25.5253% RMSE; Holm-adjusted p=0.011976.
- E2 stage-vector accuracy 0.4467; downstream NULL false-admission 0.30.
- E3 equal-budget multiscale gain 0.0.
- E4 grid and family sensitivity completed; K route selected.
- E5 10-seed structural stability completed.
- E6 N1/N2 completed across five perturbation types and four magnitudes; no observed stage/channel/scale flips.

E1 is labelled E1_COMPATIBLE_SINGLE_TASK_EXTENSION, not native full-registry PRISM E1. Historical invalid Tanks E2-E6 outputs are superseded and excluded.

## CZ

Frozen R5 evidence: E1 active route A only for Rod_1_to_Rod_2:dynamic; E2 recovery 0.650; E2 false admission 0.10625; E3 mean multiscale gain -3.9978%; E5 118 rows; E6 N1/N2 160 rows each.
