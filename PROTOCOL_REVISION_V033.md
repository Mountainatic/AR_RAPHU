# Protocol Revision v0.3.3

Revision:
v0.3.2 -> v0.3.3

Frozen v0.3.2 facts:
- targeted tests passed;
- R1 passed;
- E0U passed;
- E1A stopped at representation;
- 32x24 worst core NRMSE = 0.050123894870888544;
- the identity-reference ratio became worse as the amplitude basis improved;
- lag=40 was conditionally skipped when no amplitude candidate passed.

Repairs:
1. remove identity error ratios from all pass/fail rules;
2. compute weighted orthogonal marginal and joint projection errors;
3. run all lag x amplitude combinations unconditionally;
4. separate predictive, structural, and mother-reference resolutions;
5. separate strong rank-2 gates from weak rank-2 sensitivity tests.

Execution boundary:
- preserve every v0.3.2 result as read-only evidence;
- run E1B before every capacity experiment;
- after E1B, follow E2A0 -> E2A_M_SPACE -> E2A_S_SPACE ->
  E2A_P_NAT -> E2A_P_PERM and stop at the first frozen failure;
- generate the v0.3.3 decision and package, then pause;
- do not implement or run E2B/E3 in this revision.
