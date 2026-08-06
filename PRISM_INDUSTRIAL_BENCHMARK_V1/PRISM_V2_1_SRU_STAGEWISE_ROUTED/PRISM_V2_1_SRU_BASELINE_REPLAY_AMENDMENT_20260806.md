# PRISM v2.1 SRU baseline replay amendment

> Status: `FROZEN_USER_AUTHORIZED_BEFORE_BASELINE_REPLAY`
> Date: 2026-08-06
> Scope: baseline provenance and execution order only

This amendment records the user's explicit replacement of the baseline reuse
rule in the original v2.1 plan. It does not alter the signed theory, numerical
model grids, target heads, sample IDs, or v2.1 K/C/W/A/J selection rules.

## Frozen replacement

The historical sample-level baseline parquet cache does not exist. The
execution must therefore:

1. stop searching for, recovering, or requiring historical baseline parquet;
2. run a new `B0` stage from the frozen C2--C6 code and the unchanged
   `configs/cpu_model_freeze_v1.json` and `configs/c6_full_final_v2.json`;
3. restrict B0 to `SRU_H2S__H5__W1` and `SRU_SO2__H5__W1` with their registered
   input-only and dynamic information sets;
4. run no OOD split, no other dataset, and no private CZ data;
5. materialize new validation and test per-sample parquet with immutable sample
   IDs, `entity_id`, `origin`, `y_true`, and `y_pred`;
6. freeze every prediction SHA256 before E0;
7. run v2.1 and paired statistics only against that internal replay inventory.

The automatic order is:

```text
B0 -> E0 -> E1 -> E2-K -> E2-C -> E3-W -> E4-A -> E5-Joint
   -> E6 -> E7 -> E8
```

## Baseline-only test access exception

B0 may read the two registered SRU test partitions solely to final-fit the
already frozen baselines and write opaque test prediction hashes. B0 must not:

- calculate or persist test metrics;
- expose test loss, ranking, or model comparisons to any v2.1 development
  selector;
- alter a baseline contract after test generation;
- run in the same subprocess as E0--E6;
- make v2.1 model test access legal before E6.

The original E6/E7 guard remains binding for every v2.1 model. Manifests must
distinguish `baseline_replay_test_accessed=true` from
`v21_candidate_test_accessed=false`.

## Failure handling

An unavailable historical cache is neither searched for nor treated as a
blocker. A newly reproduced baseline failure is retained under the frozen C6
failure policy and is never replaced by an aggregate metric or fabricated
prediction. Only models with aligned validation and test per-sample artifacts
enter paired statistics.
