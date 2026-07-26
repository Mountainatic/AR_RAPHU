# Invalid SPACE Excitation Audit

Status: `INVALID_IMPLEMENTATION_AUDIT`

The first E2A-M-SPACE pre-run used a one-dimensional scrambled Sobol stream
as if it were a chronological scalar trajectory, then formed 64-lag windows
from adjacent stream values.

This did not implement the v0.3.3 SPACE identification contract. The resulting
1792-column mother-space designs had effective rank only about 30--36 and
condition numbers about 2.8e4--1.0e6. Contribution prediction and KKT residuals
were excellent, but full-surface recovery was not identifiable; AR-S4U surface
NRMSE reached 0.22--0.50.

The run is retained only as an implementation audit. It must not be used for
the v0.3.3 mother-space pass/fail decision.

Repair:

1. generate scrambled Sobol samples directly in the 64-dimensional history
   cube over the frozen core amplitude domain;
2. preserve 20,000 samples, the frozen seed offset, 70/15/15 split, FP64
   solver, smoothing grid, validation-only selection, and all gates;
3. rerun E2A-M-SPACE in the formal result path.
