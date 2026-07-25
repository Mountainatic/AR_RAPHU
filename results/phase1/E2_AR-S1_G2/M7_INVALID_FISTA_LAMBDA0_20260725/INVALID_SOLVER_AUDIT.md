# INVALID M7 solver audit

Status: `FAILED`

The first formal M7 batch used monotone restarted FISTA for all smoothing
weights. Twelve zero-smoothing candidates with larger spline grids reached the
100,000-iteration cap without satisfying the jointly frozen relative-step and
KKT convergence criteria. Although none of those approximate fits determined
the reported one-SE choice, the batch-level validation selection and test
aggregation are invalid and must not be used as scientific evidence.

The entire first M7 result namespace and its job records were preserved under
the `_INVALID_FISTA_LAMBDA0_20260725` suffix. The corrected run uses a
rank-revealing FP64 least-squares solve (`gelsd`) only for the exactly
unpenalized candidate and retains the audited V20 FISTA solve for positive
smoothing weights.
