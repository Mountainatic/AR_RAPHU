# PRISM Industrial Benchmark V1 strict restart

This directory is governed, in descending order of authority, by:

1. `PRISM_INDUSTRIAL_BENCHMARK_V1_MASTER_PROTOCOL.md`;
2. `PRISM_INDUSTRIAL_BENCHMARK_V1_CPU_PLAN.md` for C0--C7;
3. `PRISM_Theory_v1_3_Theory_Only.md` for model semantics;
4. `PRISM_INDUSTRIAL_BENCHMARK_V1_EXPERIMENT_MATRIX.csv`.

The prior PRISM CPU implementations and all their derived results were deleted on
2026-08-01 after a protocol audit. They are invalid and must not be copied,
imported, or treated as evidence. The only retained experimental inputs are the
read-only raw sources rooted at `/root/autodl-tmp/PRISM_DATASETS_V1/raw_sources`.

## Non-negotiable implementation contracts

- Work strictly in C0 -> C1 -> C2 -> C3 -> C4 -> C5 -> C6 order. C7 is not
  applicable until an independently valid GPU result package exists.
- Do not start model fitting until all five Stage-0 dataset freeze decisions pass.
- The target is the registered change
  `mean(y[t+h:t+h+W]) - mean(y[t-W0:t])`, normally with `W0=W`; it is never the
  unanchored future level.
- Every model compared within a task uses the same immutable sample IDs, split,
  target, information set, and purge.
- Input-only and dynamic leaderboards remain separate.
- The Urysohn ladder is exactly nested:
  exact-zero -> linear distributed lag -> rank-1 -> per-channel adaptive rank-R
  -> full finite Urysohn. A polynomial surrogate or post-hoc label is forbidden.
- Rank is selected independently for every `(channel, target_head)` from
  validation only. It is never a rank of the AR branch.
- Physics-First uses rolling OOF physical residuals, the maturity condition
  `s+h+W<=t`, an exact-zero residual model, and a permanently frozen K layer.
- K-Joint AR jointly optimizes a true Urysohn input layer and a target-head-specific
  state layer. Ordinary ridge ARX must never be relabeled as K-Joint AR.
- All basis knots, scalers, channel profiles, penalties, support and ranks are fit
  on outer-training only. Test data is read once after freeze.
- FP64 is mandatory for physical operators, system identification, metrics,
  bootstrap, KKT residuals and numerical certificates.
- Failed datasets, exact-zero channels, solver failures and negative directions
  remain in the record.
- Raw datasets must never be committed or included in result bundles.

## Missing specification gate

If a choice can materially alter results and is not fixed by the four governing
documents, record it in `docs/IMPLEMENTATION_FREEZE_GATE.md`. Do not silently
invent a default. C0 metadata inspection may continue, but C1 and model code may
not cross an unresolved gate.

## Change control

- Commit the protocol and Stage-0 contract before model implementation.
- Commit before any major model-family change.
- Every requirement must map to code, tests, and an output artifact in
  `docs/IMPLEMENTATION_TRACEABILITY.md`.
- Generated artifacts record commit SHA, protocol hash, raw-data hashes and the
  exact uv-managed interpreter path.

