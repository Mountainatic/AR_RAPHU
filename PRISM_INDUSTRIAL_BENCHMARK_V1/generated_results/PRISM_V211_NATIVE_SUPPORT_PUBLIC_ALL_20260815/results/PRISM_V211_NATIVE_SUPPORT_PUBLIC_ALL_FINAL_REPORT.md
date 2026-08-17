# PRISM v2.1.1 Native Support Public-All Final Report

Evidence class: `POST_LOCKBOX_MATERIALIZATION_REPAIR_WITH_FROZEN_DEVELOPMENT_AND_VALIDATED_PARTIAL_ARTIFACT_REUSE`.

The first lockbox access ended in a final-materialization runtime failure. This result reuses the unchanged frozen development artifacts after a code-equivalence and SHA256 audit, applies only the accepted materialization repair, and records 6 lockbox access attempts. The final resume reused 129 baseline test prediction artifacts only after per-artifact validation. No test result was used for reselection.

## Scope

Five public datasets and seven primary heads were evaluated with the frozen primary views. GPU baselines and multihorizon scale sweeps were out of scope.

## Primary heads

| Task | PF status | PF test MSE | Joint status | Joint test MSE | Joint vs PF |
|---|---:|---:|---:|---:|---:|
| DEB_C4 | PASS | 0.0016070561613727547 | JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT | nan | nan |
| METRO_OIL20 | PASS | 15.960861187750375 | PASS | 13.502372806314197 | 0.1540323139532731 |
| METRO_P60 | PASS | 0.043018553350118505 | PASS | 0.04621703682914463 | -0.07435125614277105 |
| PMSM_PM5 | PASS | 16.994653220318785 | JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT | nan | nan |
| SRU_H2S | PASS | 0.0030636911230938813 | JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT | nan | nan |
| SRU_SO2 | PASS | 0.002199785678733274 | JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT | nan | nan |
| TEP_G12 | PASS | 0.31340472523243895 | JOINT_STABILITY_REGISTERED_STABILITY_CONTROLS_INSUFFICIENT | nan | nan |

## Leaderboards

Input-only and dynamic leaderboards are separate. Test ranking uses the frozen task-level common support; extra native coverage is reported separately and is not used for ranking.

Top rows are descriptive test outcomes after the global freeze; they do not alter frozen selections.

## Native Support

The final audit contains 99 channel rows. Reclaimed rows are support-efficiency measurements and are not interpreted as direct prediction improvements.

## Statistics

Paired moving-block bootstrap uses 500 replicates with fixed seeds and finite-sample p-value correction. Holm correction is applied within registered comparison families.

## OOD

OOD is reported only for registered TEP/Metro and other available OOD views. OOD residual-state construction reuses frozen test residuals where required by the registered causal protocol.

## Interpretation

The results describe predictive contribution, structured response evidence, conditional novelty, and module activation or degradation. They do not prove causality or mechanism.

The most direct residual risk is that the frozen runner did not persist peak-RAM and separate fit/prediction timing counters; resource output marks those fields as not recorded.
