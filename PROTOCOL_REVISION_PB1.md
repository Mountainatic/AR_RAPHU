# OPS-UOI Public Benchmark PB0/PB1 protocol freeze

Date: 2026-07-27

## Immutable starting evidence

- Source branch: `ps-ar-raphu-v034-adaptive-rank-profile`
- Source commit: `e07f0d5598a82c73036f5b7ad2f5686e76a2cf31`
- Evidence archive:
  `SPECTRAL_PS_AR_RAPHU_V034_RANK_PROFILE_RESULTS.zip`
- Evidence SHA256:
  `8cb233c646362fb6a65e732c7f135dff4043ea9d672af5ffa71d7dbe2112c8e3`
- New branch: `public-benchmark-pb1`

The v0.3.4 configuration, results and scientific decision remain read-only.
PB1 does not rerun or retune the v0.3.3/v0.3.4 synthetic experiments.

## Authorized scope

This branch implements only:

1. PB0 source, license, hash, chronology and split auditing;
2. the common `DynamicDataset` contract;
3. sequence-aware causal windowing and train-only preprocessing;
4. official loaders for PWH, WH process-noise, Cascaded Tanks and Silverbox;
5. PB1 smoke, development, protocol freeze and confirmation;
6. PB1 reports and the frozen return package.

PB2, PB3, PB4 and CZ are outside this branch. Private CZ data must not be
read, copied or included in PB1 artifacts.

## Mathematical and claim boundaries

- The existing FP64 spectral normal-equation, Gram whitening and tail-rank
  definitions are unchanged.
- Public data may report predictive SVD rank and stability intervals.
- Structural rank is forbidden unless a K-level certificate explicitly
  permits it.
- Public datasets have no truth-rank or truth-kernel metrics.
- Fixed rank-2 remains a comparator, never a universal hypothesis.
- Full, rank-1, rank-2 and adaptive models share the same split, scaler and
  test-access contract.

## Selection and test access

The fixed order is:

1. freeze the official or outer test records;
2. select history on development data;
3. select resolution;
4. select regularization;
5. fit the full kernel;
6. select predictive SVD rank on validation data;
7. freeze the model;
8. access official test once;
9. never revise the primary model after test.

No window may cross `sequence_id`. No future X or Y may enter a direct
forecast. Scalers, domains, bases and data-driven preprocessing are fit on
training data only. OOD values are flagged and never silently clipped.

## Runtime

PB1 reference fitting is CPU FP64. Parallelism is at
dataset/model/horizon/seed task level. CUDA and mixed precision are excluded
from the first PB1 package.

## Gates

The implementation gate requires:

```text
PB0_SOURCE_AUDIT_PASS
ALL_LEGACY_REGRESSION_TESTS_PASS
ALL_PB1_LOADERS_PASS
ALL_PB1_SMOKE_PASS
PB1_PROTOCOL_FROZEN
OFFICIAL_TEST_ACCESSED_ONCE
PB1_RESULT_PACKAGE_VALID
```

The scientific gate additionally requires at least one clear full-kernel
gain over rank-1, adaptive-rank performance within the frozen predictive
budget, preserved process-noise records, separate OOD reporting and retained
negative results. Failing the scientific gate yields
`STOP_AND_REVISE_MODEL`, not automatic PB2 execution.

## Migration audit

Before PB1 changes:

- the v0.3.4 archive hash and CRC were verified;
- 151 repository tests passed;
- seven legacy Phase-0 tests could not run because the historical private-CZ
  manifest snapshot is absent from the public worktree;
- root-level pytest also discovered duplicate tests inside the untracked
  `return_v034/` package.

The PB1 branch must make test discovery deterministic and classify the absent
private-CZ snapshot without reading or reconstructing private data.
