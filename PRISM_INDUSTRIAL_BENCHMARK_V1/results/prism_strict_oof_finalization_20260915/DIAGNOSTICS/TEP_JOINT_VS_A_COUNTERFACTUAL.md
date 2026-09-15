# TEP record-time Joint versus standalone A diagnostic

All 56 registered Joint candidates plus frozen standalone A were evaluated on the same validation support and on common causal folds 1--3. Fold 0 is excluded from the OOF comparison because standalone A has no preceding causal fit fold. The full 114-row table is in the companion CSV.

| comparison_set | scope | route | k_representation | predictive_eta | outer_oof_objective | Delta_RMSE | Delta_R2 | Level_RMSE | Level_R2 | parameter_count | input_block_contribution | A_block_contribution | support_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOF_COMMON_FOLDS_1_3 | STANDALONE_A | A |  |  | 0.09724449093 | 0.3118404896 | 0.03682274316 | 0.3118404896 | 0.722251425 | 5 | 0.0002566622633 | 0.003505892377 | 3a604de324fb150be04724a6acbd8db7ea51b24abc931fa4a127c8ad46fb7d7a |
| OOF_COMMON_FOLDS_1_3 | REGISTERED_JOINT_CANDIDATE | J_K | FULL_BASIS | 0.1 | 0.1007855195 | 0.3174673519 | 0.001749926445 | 0.3174673519 | 0.7121375806 | 289 | 0.0001915655477 | 0 | 3a604de324fb150be04724a6acbd8db7ea51b24abc931fa4a127c8ad46fb7d7a |
| OOF_COMMON_FOLDS_1_3 | REGISTERED_JOINT_CANDIDATE | J_KA | FULL_BASIS | 0.01 | 0.08082637239 | 0.2842997932 | 0.1994392392 | 0.2842997932 | 0.7691446627 | 297 | 0.005476710779 | 0.0256769464 | 3a604de324fb150be04724a6acbd8db7ea51b24abc931fa4a127c8ad46fb7d7a |
| OOF_COMMON_FOLDS_1_3 | REGISTERED_JOINT_CANDIDATE | J_KW | FULL_BASIS | 1 | 0.1007955847 | 0.3174832038 | 0.001650233948 | 0.3174832038 | 0.7121088326 | 297 | 0.0001625253295 | 0 | 3a604de324fb150be04724a6acbd8db7ea51b24abc931fa4a127c8ad46fb7d7a |
| OOF_COMMON_FOLDS_1_3 | REGISTERED_JOINT_CANDIDATE | J_KWA | FULL_BASIS | 0.01 | 0.08084377776 | 0.2843304024 | 0.1992668443 | 0.2843304024 | 0.7690949497 | 305 | 0.005544993256 | 0.02566617782 | 3a604de324fb150be04724a6acbd8db7ea51b24abc931fa4a127c8ad46fb7d7a |

## Mismatch audit

- selection_objective_mismatch: `true`
- calibration_mismatch: `false`
- calibration_audit: `distinct registered estimators; prediction replay and shared-error identity passed`
- support_mismatch: `false`
- delta_vs_level_objective_mismatch: `false`
- common_folds: `[1, 2, 3]`
- fold_0_exclusion: `standalone a has no preceding causal fit fold`

This is development-only report evidence; it did not change the selector, objective, calibration, support, or admission semantics.
