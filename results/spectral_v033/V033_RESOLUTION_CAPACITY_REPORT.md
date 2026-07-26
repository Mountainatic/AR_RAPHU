# Spectral PS-AR-RAPHU v0.3.3 Resolution Capacity Report

## Executive result

The v0.3.3 representation repair succeeded, implementation closure succeeded,
and the mother-space estimator passed. The structural-space experiment stopped
at its pre-registered rank-truncation gate:

```text
PRIMARY_LIMITATION: STRUCTURAL_SPACE_CAPACITY
NEXT_ALLOWED_STAGE: STOP_STRUCTURAL_SPACE_CAPACITY
```

No predictive NAT/PERM experiment and no E2B/E3 experiment was run.

## E1B representation certificate

All 1,440 pre-registered lag-amplitude combinations were evaluated
unconditionally. The frozen roles all passed:

| Role | Resolution | Worst core joint NRMSE | Worst fit joint NRMSE |
|---|---:|---:|---:|
| Predictive | 32 x 28 | 0.0299971 | 0.0189913 |
| Structural | 48 x 28 | 0.0252387 | 0.0132533 |
| Mother | 64 identity x 28 | 0.0252020 | 0.0132277 |

The strong-rank representation operator/gap checks also passed. The obsolete
identity-error ratio was not used.

## E2A0 implementation closure

```text
maximum truth replay error:                 8.8818e-16
maximum matrix/direct projected error:      3.3029e-15
projected-target prediction relative error: 1.1138e-12
direct KKT relative residual:               3.5144e-16
PCG KKT relative residual:                  4.7442e-11
direct/PCG prediction relative difference:  5.6086e-11
```

All frozen closure tolerances passed.

## E2A-M-SPACE mother-space capacity

All 12 scenario-variable groups passed with 5/5 seeds. Across 60 tasks:

```text
validation contribution R2: 0.9992973 .. 1.0000000
empirical operator NRMSE:    9.02e-08 .. 0.02651
core surface NRMSE:          3.36e-05 .. 0.02546
KKT relative residual:       3.55e-16 .. 7.82e-16
```

This supports the narrow conclusion that the 64-lag identity mother space,
28-function amplitude basis, and frozen convex estimator have single-variable
SPACE capacity for the tested truth family.

## Invalid one-dimensional Sobol pre-run

The first mother-space pre-run treated a one-dimensional Sobol stream as a
chronological trajectory. Its 1792-column designs had effective rank only about
30--36, so it did not fill the lag-history cube. That run is isolated under
`invalid_runs/` and is not used for any scientific decision.

The repaired formal run used 20,000 scrambled-Sobol points directly in the
64-dimensional history cube, preserving the frozen core domain, seed offset,
70/15/15 split, smoothing grid, FP64 solver, and gates.

## E2A-S-SPACE structural-space capacity

S1, S2, and S3 all passed for all three variables and all five seeds. Their
rank-1 preservation and weak/strong rank-2 checks behaved as intended.

AR-S4U also passed full-space contribution and surface recovery:

```text
validation contribution R2: at least 0.9993073
empirical operator NRMSE:    at most 0.02632
core surface NRMSE:          at most 0.02545
```

However, every AR-S4U variable failed the frozen rank-2-to-full equivalence
gate. Its rank-2 validation MSE was about 0.00434--0.00623, while the full
48x28 validation MSE was about 2.53e-06--3.17e-05. Rank-2 captured only
0.621--0.908 of the rank-1-to-full reducible gap, and could not remain within
1.05 times the full validation MSE.

Therefore the result does not show failure of the full structural surface.
It shows that the frozen rule cannot compress the higher-order AR-S4U surface
to rank 2 while retaining full-model accuracy. Under the execution contract
this is still a structural-capacity stop, and no later experiment is unlocked.

## Conclusion boundary

Supported:

- the repaired weighted representation certificate;
- mother-space single-variable SPACE capacity;
- 48x28 full-surface recovery for the tested rows;
- rank-1 preservation for S1/S2;
- rank-2 recovery for the frozen S3 strong-rank rows.

Not supported or not run:

- universal rank-2 compression of amplitude-dependent lag surfaces;
- predictive NAT/PERM capacity;
- multivariable E2B capacity;
- double-residual E3 estimation;
- support/rank inference beyond the completed capacity diagnostics;
- deployment claims.
