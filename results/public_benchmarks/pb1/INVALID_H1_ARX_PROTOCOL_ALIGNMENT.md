# INVALID H1 ARX protocol alignment audit

Status: `FAILED`

The development artifacts originally written under:

- `pwh/development/H1_ARX/history_selection.json`
- `whpn/development/H1_ARX/history_selection.json`

used the companion paper's contemporaneous system-identification convention
`u_t -> y_t`. PB1 instead freezes information at forecast origin `t` and uses
`X[<=t], y[<=t] -> y[t+1]`; future `X[t+1]` is forbidden.

The original `(nx, ny)=(18,19)` and `(19,15)` selections are retained only as
invalid audit evidence. They must not enter selection, aggregation, protocol
freeze, confirmation, or reporting.

Corrected results are written to the separate
`H1_ARX_NO_FUTURE_X/` namespace and never overwrite these files.
