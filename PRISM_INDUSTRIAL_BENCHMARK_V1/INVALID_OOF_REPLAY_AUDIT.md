# Invalid OOF replay-audit pre-run

Status: `INVALID_IMPLEMENTATION_AUDIT_PRE_RUN_ISOLATED`.

The high-concurrency launch completed M0/M1 and entered M2. It was terminated
before M2 completion or any development freeze after 2 of 22 completed K
channels were incorrectly marked `SOLVER_FAILED_RETAINED`. Both failures were
caused by an implementation-only `1e-12` equality assertion between a selection
loss and a later non-selecting ALS replay for TP2. The assertion is not part of
the frozen scientific contract. No candidate, threshold, initialization,
budget, data row, or selection rule failed.

The correction records the replay difference, binds the final fold loss to the
materialized OOF prediction, and never feeds the replay back into selection.
No M2 summary, development freeze, test access, or OOD access from this launch
is valid evidence.
