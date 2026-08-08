# PRISM v2.2 Joint Stability Pre-Run Review

Status: `PASS`

This review is completed before formal Metro M5 execution. Formal execution is forbidden unless every item below is PASS.

| Requirement | Review result | Evidence |
|---|---|---|
| M2-M4 estimator files unchanged | PASS | `V22_M2_M4_ARTIFACT_REUSE_AUDIT.json`; all source/current SHA256 pairs equal |
| Original four-fold `T_i -> V_i` protocol retained | PASS_CODE_REVIEW | `registered_joint_inner_fold_frames()` and `audit_joint_fold_protocol()` reused |
| Test/OOD guards unchanged and no access | PASS_CODE_REVIEW | M5 loads train and validation only; result flags remain false |
| Input-path gate thresholds unchanged | PASS_CODE_REVIEW | inherited `_gate_config()` and `input_path_preservation_gate()` |
| Eta grid matches theory/config | PASS_CODE_REVIEW | `[0,1e-5,1e-4,1e-3,1e-2,1e-1,1]` |
| Compressed/full raw K support identical | PASS_CODE_REVIEW | both taken from one `fit_physical_features()` call and checked against frozen active K |
| Numerical and predictive ridge separated | PASS_CODE_REVIEW | bare alpha selected by certificates; predictive penalty is `n_fit * eta * I` |
| Legacy corrected v2.1.2 anchor reproduced | PASS_PRE_RUN | exact solver regression passed; formal M5 rechecks the real four-fold artifact before any v2.2 candidate scan |
| Candidate binding includes route/representation/alpha/eta | PASS_CODE_REVIEW | `V22Candidate` descriptor and stable candidate ID |
| Formal route set unchanged | PASS_CODE_REVIEW | exactly `J_K/J_KW/J_KA/J_KWA`; AR-only and K-zero rejected |
| K/W/A matrices constructed once per fold | PASS_CODE_REVIEW | sufficient statistics reused for ordered candidate solves |
| M7 compatibility does not reapply predictive ridge | PASS_CODE_REVIEW | frozen coefficient contract used directly for prediction |

Full regression status is `177 passed, 1 skipped`. The artifact reuse audit is PASS, including unchanged shared-data aggregate, K/C/W/A RESULT files, C/W OOF files, unchanged K/C/W/A estimator sources, and `test_accessed=false`, `ood_accessed=false`. The formal M5 implementation treats real-artifact legacy-anchor mismatch as a hard stop before evaluating the v2.2 predictive path. M7/M8 and test/OOD remain out of scope.
