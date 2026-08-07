# Invalid pre-run audit: M4 ablation OOF merge omission

Date: 2026-08-07 (Asia/Shanghai)

The first fork-parallel M4 pre-run is excluded from formal scientific evidence.
Its A candidate calculations completed without OOM, but both views ended as
`SOLVER_FAILED_RETAINED` during route materialization.

Root cause: W correctly materialized `delta_w_ablation_oof` and
`physical_w_ablation_oof`, while A's OOF merge selected neither column before
the frozen KCWA route refit. Both views therefore raised the same `KeyError`.
This was a deterministic implementation omission, not a model, selection, or
scientific failure.

- M0--M3 remained valid and unchanged.
- M4 did not complete and no M4 decision was produced.
- Development was not frozen.
- Candidate test and OOD data were not accessed.
- The cgroup recorded zero OOM and zero OOM kills.

The repair centralizes the W-to-A OOF merge, explicitly requires both registered
W routes, and adds a regression test proving the ablation columns survive the
merge. The frozen M1 suite passes with 69 tests after the repair.
