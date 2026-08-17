# Native Support Summary

Channel-level reclaimed training rows (non-negative sum): **1093139**.
This is a support-efficiency statistic, not a causal or predictive improvement claim.

## Highest recovery channels

- `torque`: 146188 rows
- `u_q`: 146188 rows
- `u_d`: 146188 rows
- `i_q`: 146188 rows
- `xmv_1`: 102400 rows
- `xmeas_27`: 102400 rows
- `motor_speed`: 96667 rows
- `i_d`: 96667 rows
- `coolant`: 96667 rows
- `DV_pressure`: 1512 rows

## Selection and route changes

Historical per-channel selection metadata was not available in the frozen run namespace; selection changes are reported as `NOT_AVAILABLE` rather than inferred from aggregate historical metrics.
PF and Joint route changes are read from the global freeze records. Test-direction statements are descriptive correlations only.
