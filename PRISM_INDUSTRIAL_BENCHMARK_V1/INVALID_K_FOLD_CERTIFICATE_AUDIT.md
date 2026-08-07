# Invalid K inner-fold certificate pre-run audit

Status: `INVALID_IMPLEMENTATION_AUDIT_PRE_RUN_ISOLATED`.

The launch requested 27 task workers, used one BLAS thread per process, and the
cgroup-aware scheduler resolved the pool to 23 concurrent workers. It completed
22 of 27 K `RESULT.json` files before intentional termination. All 22 had a
top-level `PASS` status, but both completed TP2 views contained `Infinity` in
fold 1 of `selection_fold_losses_before_oof_replay` while their separately
replayed, materialized OOF losses were finite.

The cause was an implementation gap in the minimal-stabilizing-ridge step: the
ridge was certified only on the full-train refit, not on every inner-fold fit.
A later independent OOF replay could therefore produce finite predictions for a
ridge whose selection-time inner-fold certificate had failed. That evidence is
not acceptable for final candidate/loss/prediction binding.

The correction keeps the registered ridge grid and its order unchanged, selects
the first ridge whose full refit and all four inner folds certify, and directly
materializes OOF predictions from those certified fold fits. It does not change
profiles, structural candidates, thresholds, row caps, data, precision,
randomness, or any scientific selection rule.

M2 did not complete, no M2 summary or development freeze was written, and no
test or OOD data was accessed. This namespace and its chain log are retained
only as invalid execution-audit evidence and must never be resumed or reported
as scientific evidence.
