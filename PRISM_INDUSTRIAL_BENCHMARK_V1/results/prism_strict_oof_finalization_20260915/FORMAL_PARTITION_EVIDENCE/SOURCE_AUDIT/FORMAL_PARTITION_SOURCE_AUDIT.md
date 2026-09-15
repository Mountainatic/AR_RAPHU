# Formal Partition Source Audit

## Verdict: A_UNIQUELY_MATERIALIZABLE / PASS

| task | source | rule | H/W | test rows | OOD rows | materialized |
|---|---|---|---|---:|---:|---|
| TEP | frozen TEP run/fault split registry; H0 strict-past C1 sample IDs | TEP_RUN_FAULT_HOLDOUT_V1; [t-L,t), D[t] excluded, Lmax=256 | H0/W1 | 16856000 | 5267500 | PASS |
| Debutanizer | frozen chronological C1 sample IDs | chronological_60_20_20_floor_boundaries with registered purge | H5/W1 | 1403 | 0 | PASS |
| SRU-H2S | frozen chronological C1 sample IDs | chronological_60_20_20_floor_boundaries with registered purge | H1/W1 | 4010 | 0 | PASS |
| SRU-SO2 | frozen chronological C1 sample IDs | chronological_60_20_20_floor_boundaries with registered purge | H1/W1 | 4010 | 0 | PASS |
| PMSM | frozen profile-holdout C1 sample IDs | PRISM_PMSM_SPLIT_V1 profile_id isolation | H600/W60 | 928044 | 0 | PASS |
| MetroPT-P60 | frozen calendar-month and fault-window C1 sample IDs | calendar month test and registered fault-window OOD with purge | H6/W1 | 2561792 | 118744 | PASS |
| MetroPT-Oil20 | frozen calendar-month and fault-window C1 sample IDs | calendar month test and registered fault-window OOD with purge | H120/W12 | 1280080 | 58284 | PASS |

TEP H0 OOD was reconstructed from the frozen task-independent OOD base membership, not from H1/W2 sample IDs. Only `entity_id` and `row_in_entity` were read, then the frozen H0 support rule was applied.

The legacy public-all inference closure remains `INCOMPATIBLE` and was not used.
