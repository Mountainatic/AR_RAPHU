# MULTISCALE-PHYSICS-AUDIT V1 Final Report

## Answer first

- Pipeline status: `COMPLETED`.
- Stage 1 completed profile-variants: `42`.
- Stage 2 confirmed linear structure: `2`.
- AR-conditional gains: `1`.
- Stable nonlinear K gains: `0`.

The raw workbook is excluded from source control and the return bundle. All claims below concern two provided rods and the registered stable segments only.

## Stage 1 scale candidates

| Task | Channel | Horizon min | Window min | History min | Pooled Q gain | S1 |
|---|---|---:|---:|---:|---:|---|
| C1__lowpass | crucible_rotation | 2 | 1 | 30 | 0.061% | S1_CANDIDATE_FAIL |
| C1__midband | crucible_rotation | 2 | 1 | 30 | 0.017% | S1_CANDIDATE_FAIL |
| C1__raw | crucible_rotation | 2 | 1 | 30 | 0.044% | S1_CANDIDATE_FAIL |
| C2__lowpass | crucible_rotation | 5 | 2 | 30 | 3.541% | S1_CANDIDATE_FAIL |
| C2__midband | crucible_rotation | 5 | 2 | 30 | 3.572% | S1_CANDIDATE_FAIL |
| C2__raw | crucible_rotation | 5 | 2 | 30 | 3.553% | S1_CANDIDATE_FAIL |
| C3__lowpass | crucible_rotation | 10 | 2 | 60 | 0.173% | S1_CANDIDATE_FAIL |
| C3__midband | crucible_rotation | 10 | 2 | 60 | 0.438% | S1_CANDIDATE_FAIL |
| C3__raw | crucible_rotation | 10 | 2 | 60 | 0.137% | S1_CANDIDATE_FAIL |
| C4__lowpass | crucible_rotation | 20 | 5 | 60 | -0.009% | S1_CANDIDATE_FAIL |
| C4__midband | crucible_rotation | 20 | 5 | 60 | 0.262% | S1_CANDIDATE_FAIL |
| C4__raw | crucible_rotation | 20 | 5 | 60 | 0.064% | S1_CANDIDATE_FAIL |
| C5__lowpass | crucible_rotation | 40 | 5 | 120 | -0.045% | S1_CANDIDATE_FAIL |
| C5__midband | crucible_rotation | 40 | 5 | 120 | -0.139% | S1_CANDIDATE_FAIL |
| C5__raw | crucible_rotation | 40 | 5 | 120 | -0.146% | S1_CANDIDATE_FAIL |
| C6__lowpass | crucible_rotation | 60 | 10 | 120 | -2.494% | S1_CANDIDATE_FAIL |
| C6__midband | crucible_rotation | 60 | 10 | 120 | -0.756% | S1_CANDIDATE_FAIL |
| C6__raw | crucible_rotation | 60 | 10 | 120 | -18.373% | S1_CANDIDATE_FAIL |
| R1__innovation | crystal_rotation | 2 | 1 | 30 | 0.299% | S1_CANDIDATE_FAIL |
| R1__raw | crystal_rotation | 2 | 1 | 30 | 0.014% | S1_CANDIDATE_FAIL |
| R2__innovation | crystal_rotation | 5 | 2 | 30 | 3.878% | S1_CANDIDATE_FAIL |
| R2__raw | crystal_rotation | 5 | 2 | 30 | 3.505% | S1_CANDIDATE_FAIL |
| R3__innovation | crystal_rotation | 10 | 2 | 60 | 1.407% | S1_CANDIDATE_PASS |
| R3__raw | crystal_rotation | 10 | 2 | 60 | 0.030% | S1_CANDIDATE_FAIL |
| R4__innovation | crystal_rotation | 20 | 5 | 60 | 2.231% | S1_CANDIDATE_PASS |
| R4__raw | crystal_rotation | 20 | 5 | 60 | -38.679% | S1_CANDIDATE_FAIL |
| R5__innovation | crystal_rotation | 40 | 5 | 120 | 2.107% | S1_CANDIDATE_PASS |
| R5__raw | crystal_rotation | 40 | 5 | 120 | -0.865% | S1_CANDIDATE_FAIL |
| R6__innovation | crystal_rotation | 60 | 10 | 120 | 0.211% | S1_CANDIDATE_FAIL |
| R6__raw | crystal_rotation | 60 | 10 | 120 | 0.096% | S1_CANDIDATE_FAIL |
| P1__raw_plus_thermal_states | heater_power | 10 | 5 | 60 | -0.015% | S1_CANDIDATE_FAIL |
| P2__raw_plus_thermal_states | heater_power | 20 | 5 | 60 | 0.016% | S1_CANDIDATE_FAIL |
| P3__raw_plus_thermal_states | heater_power | 40 | 10 | 120 | 0.018% | S1_CANDIDATE_FAIL |
| P4__raw_plus_thermal_states | heater_power | 60 | 10 | 120 | 3.686% | S1_CANDIDATE_PASS |
| P5__raw_plus_thermal_states | heater_power | 90 | 15 | 180 | -1.772% | S1_CANDIDATE_FAIL |
| P6__raw_plus_thermal_states | heater_power | 120 | 15 | 180 | 4.039% | S1_CANDIDATE_FAIL |
| L1__pc1 | joint_lift | 0.5 | 0.5 | 10 | 7.850% | S1_CANDIDATE_FAIL |
| L2__pc1 | joint_lift | 1 | 0.5 | 10 | 11.570% | S1_CANDIDATE_PASS |
| L3__pc1 | joint_lift | 2 | 1 | 20 | 17.099% | S1_CANDIDATE_PASS |
| L4__pc1 | joint_lift | 5 | 1 | 20 | 19.096% | S1_CANDIDATE_PASS |
| L5__pc1 | joint_lift | 10 | 2 | 40 | 42.146% | S1_CANDIDATE_PASS |
| L6__pc1 | joint_lift | 20 | 2 | 40 | 48.060% | S1_CANDIDATE_PASS |

## Stage 2 structural and conditional evidence

| Task | Q gain | Bootstrap P(>0) | Kernel corr | S2 | AR-conditional |
|---|---:|---:|---:|---|---|
| L6__pc1 | 48.060% | 1.000 | 0.813 | True | True |
| L5__pc1 | 42.146% | 1.000 | 0.889 | True | False |
| R4__innovation | 2.231% | 0.930 | 0.865 | False | False |
| R5__innovation | 2.107% | 0.978 | 0.683 | False | False |
| P4__raw_plus_thermal_states | 3.686% | 0.978 | -0.011 | False | False |

## Decision boundaries

- `structure evidence` means bidirectional cross-rod improvement plus the predeclared bootstrap, kernel-correlation, support-overlap, and common-support gates.
- `AR-conditional gain` compares frozen scale-matched AR plus Q against the same frozen AR.
- `nonlinear K gain` is only evaluated after Stage 2 and retains an exact zero nonlinear block.
- The 180-minute heater profile is exploratory and cannot become a confirmatory result.
- Combined model: `NOT_APPLICABLE_NO_SHARED_TARGET_PROFILE`. Profiles intentionally use different horizons, output windows, and cadences. A cross-channel combined model would require a separately predeclared shared target profile and was not selected after seeing data.

## Reproducibility

- Config SHA256: `3e437a5bc7e9f4b2002b618eb1314396cc95f0c28409ce5085e4a8914e5a947f`
- Data SHA256: `c3428966fe006572809156ee5e3f488264b8206b19b20887dcd00840bb26fbc3`
- Linear algebra certification path: CPU FP64.
- Profile failures are isolated and recorded rather than aborting the run.
