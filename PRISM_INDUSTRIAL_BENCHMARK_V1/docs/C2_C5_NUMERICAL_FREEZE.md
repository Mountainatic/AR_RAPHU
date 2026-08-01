# C2--C5 numerical freeze

Status: `FROZEN_BEFORE_PARAMETERIZED_MODEL_VALIDATION`

The user authorized Codex on 2026-08-02 to pre-register and freeze every
remaining C2--C5 numerical choice before parameterized development results are
inspected, then run the strict chain through C6, packaging, commit and push.

The canonical machine-readable contract is
`configs/cpu_model_freeze_v1.json`. Its SHA256 must be recorded by every stage.
No stage may override it from the CLI.

The final hash is recorded after the pre-run computational-feasibility audit.
That audit may reduce a declared maximum budget or replace a Cartesian scan by
a result-independent staged scan, but cannot add candidates or inspect model
validation results.

Final frozen SHA256 after that audit:
`d7d90c9126d0a70e2cb3a6e859787a8bfb70da79dd5ba963573fa7b0842b4a4b`.

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
