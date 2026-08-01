# C2--C5 numerical freeze

Status: `FROZEN_BEFORE_DEVELOPMENT_RESULT_INSPECTION`

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
`14cefef2b0b4307ca208d0f0d43b50146de9edf78e93c59d67bf6fdd5d9cf65a`.

The unattended C2 computation began against full-file hash
`d7d90c9126d0a70e2cb3a6e859787a8bfb70da79dd5ba963573fa7b0842b4a4b`.
Before any development metric was inspected, C5 residual-history realization
and C6 result-independent finalist fields were completed. The canonical
`{selection,c2}` subtree hash is identical in both files:
`15d3665009caf12fb2102964bf24ea8b7b2ab7b7c973c3b7481ca4fa77100244`.
Thus the already-computed C2 candidates were not changed or selected from their
results; downstream stages use the final full-file hash above.

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
