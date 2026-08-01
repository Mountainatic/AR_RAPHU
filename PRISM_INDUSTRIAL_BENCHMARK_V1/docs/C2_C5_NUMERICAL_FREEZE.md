# C2--C5 numerical freeze

Status: `FROZEN_BEFORE_PARAMETERIZED_MODEL_VALIDATION`

The user authorized Codex on 2026-08-02 to pre-register and freeze every
remaining C2--C5 numerical choice before parameterized development results are
inspected, then run the strict chain through C6, packaging, commit and push.

The canonical machine-readable contract is
`configs/cpu_model_freeze_v1.json`. Its SHA256 must be recorded by every stage.
No stage may override it from the CLI.

Frozen SHA256:
`9af20e322905415263717adaab1befde6c61f7162632b1268ae20f095b68b889`.

Key safeguards are:

- four train-only folds and one-SE selection across folds;
- deterministic SHA256 subsampling where a solver cannot use every training
  row, while predictions cover every immutable evaluation row;
- staged representation/profile selection followed by staged penalty selection;
- exact-zero is a formal candidate for physical channels and residual/state
  additions;
- rank is selected per `(channel, target_head)`, never for the AR branch;
- C4 uses a truly nested zero/linear/rank/full finite-Urysohn ladder;
- Physics-First uses rolling OOF physical residuals and the exact maturity rule;
- K-Joint AR is a joint solve for a true Urysohn basis plus a head-specific
  state basis;
- all physical, system-identification, certificate, metric and bootstrap
  calculations use FP64;
- test and OOD remain closed until a final freeze manifest is atomically written
  at the start of C6.

The freeze is result-independent. Existing Mean/Persistence development metrics
were not used to choose any grid, threshold, basis, rank or solver option.
