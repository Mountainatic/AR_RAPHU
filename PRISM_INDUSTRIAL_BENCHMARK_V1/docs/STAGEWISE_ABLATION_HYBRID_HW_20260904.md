# PRISM v2.1.1 stagewise ablation: hybrid h/w protocol

This amendment freezes the submission-critical stagewise ablation before its
execution. R2 amends only the runtime configuration source after implementation
preflight exposed a pre-existing checksum drift; no task, metric, selection
rule, numerical threshold, or test result was changed. The code base is release tag
`prism-v2-1-1-public5-cz-neural3-level-r2-20260901`, commit
`be7557933f4a2ad11c7655c04ce031f524611cb0`. The tag is not modified; execution
uses a new branch and namespace.

## Scope boundary

Only TEP is replaced by the strict-past H0/W1 nowcast. Debutanizer, both SRU
heads, PMSM, both MetroPT heads, and CZ retain their registered future tasks.
The main table therefore compares heterogeneous industrial estimation tasks and
must not be described as a homogeneous forecast leaderboard.

The exact machine-readable task matrix is
`configs/stagewise_ablation_hybrid_hw_20260904.json`. No `all-nowcast` registry
may be used in this experiment.

Numerical K/C/W/A/J settings are loaded from the validated Public-All frozen
contract in this release. The older representative descriptor is not used as a
launcher contract because its bound CZ-contract checksum already drifts at the
release base; the checksum is not edited or bypassed.

## Stage definitions

The mandatory columns are:

1. `K`: the best active K channel selected inside C on development data,
   replayed on the same assembly common support. It is not an equal-weight sum.
2. `K+C`: frozen C fusion.
3. `K+C+DELTA_W`: frozen W correction; rejected W materializes as identity.
4. `K+C+DELTA_W+A`: Physics-First; rejected A materializes as exact zero.
5. `J`: the registered predictive Joint route. A development-rejected J is
   reported as unsupported and is never replaced by an AR-only fallback.

`K+C+A` remains a supplementary non-adjacent ablation and is not substituted
for the required stagewise ladder.

## Metrics and inference

Level RMSE, MAE, and R2 are primary. Delta RMSE, MAE, R2, and persistence skill
are retained for dynamic views, especially TEP H0. Adjacent RMSE gains are
defined with positive values indicating improvement. Uncertainty uses 500
paired moving-block bootstrap replicates, with block length fixed before test
inference from the `K+C+DELTA_W+A` development-validation residuals only. The
deterministic rule selects the first lag whose absolute within-entity ACF is at
most `1.96/sqrt(N)`, or lag 256 (or the longest available lag) if none qualifies.
Blocks never cross entity boundaries. Holm correction covers the 27 main
adjacent-stage comparisons.

The figure is a three-panel horizontal dot-and-interval comparison (`C`,
`DELTA_W`, `A`). It uses relative RMSE reduction for cross-task readability and
a visible zero reference; exact raw-unit RMSE gains remain in the table and CSV.
The palette is limited to blue, gold, and orange, while position, panel labels,
and signed values preserve meaning without color.

This is retrospective ablation evidence because relevant test supports have
already been accessed in earlier releases. It must not be relabeled as a fresh
untouched confirmatory experiment.

## Privacy and storage

CZ public artifacts contain aggregate metrics, uncertainty summaries, and
status only. Raw CZ rows, prediction rows, workbook content, and identifiers are
excluded. Large run artifacts live under `/root/autodl-tmp`; execution stops if
free data-disk space falls below 15 GiB.
