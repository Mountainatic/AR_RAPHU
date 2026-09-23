# CZ raw-2s E1–E6 validation

Status: **PASS**

Protocol: raw 2 s, L256, h=4 (8 s), W/W0=1/1, `D[t+3]-D[t-1]`.

## Checks

- PASS: `run_completed`
- PASS: `protocol_exact`
- PASS: `git_exact`
- PASS: `raw_absent`
- PASS: `row_counts_exact`
- PASS: `all_csv_numeric_finite`
- PASS: `e6_n1_zero_closes_to_formal`
- PASS: `e6_levels_exact`
- PASS: `e6_raw_prefusion_provenance`
- PASS: `e3_equal_budget`
- PASS: `target_audits_pass`
- PASS: `direction_support_consistent`
- PASS: `prediction_rows_match`
- PASS: `worktree_still_clean`

## Row counts

- `E1_STAGEWISE/formal_metrics.csv`: 4
- `E2_CZ_INPUT_SEMISYNTHETIC_RECOVERY/per_seed_stage_recovery.csv`: 320
- `E3_MULTISCALE/equal_budget_multiscale.csv`: 40
- `E4_CANDIDATE_SENSITIVITY/all_variants.csv`: 14
- `E5_STRUCTURAL_STABILITY/stability_evidence.csv`: 118
- `E6_MEASUREMENT_ROBUSTNESS/N1_frozen_per_seed.csv`: 200
- `E6_MEASUREMENT_ROBUSTNESS/N2_reidentified_per_seed.csv`: 200

## Scientific gate

- Verdict: **STOP**
- E3–E6: computed but exploratory; not gate-authorized confirmatory evidence.
- Details: `PHASE_A_GATE.json` and `CLAIM_IMPACT_ASSESSMENT.md`.
