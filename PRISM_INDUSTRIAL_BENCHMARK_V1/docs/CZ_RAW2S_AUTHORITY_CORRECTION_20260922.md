# CZ raw-2s H4 authority correction

Status: `AUTHORITY_H4_RERUN_PARTIAL`

The aggregate results previously published under
`CZ_RAW2S_H4_CORRECTED_R2` were produced by a custom rolling-statistics,
PCA and Ridge proxy.  That runner reused the strict nested-OOF selector but
did not call the authoritative PRISM K/C/W/A/Joint implementations.  Those
E1--E6 results are invalid for claims about authoritative PRISM.

The corrected entry point is `scripts/run_cz_raw2s_e1_e6.py`.  It contains no
custom PRISM feature or estimator implementation and delegates execution to
`scripts/run_independent_extension_20260825.py` with the frozen H4 config
`configs/cz_raw2s_h4_authoritative_e1_e6_20260922.json`.

Before any data access, the runner verifies that:

- authority commit `2ee6273b8f915cbcdff2f46d56bc80047ddae4a7` is an ancestor;
- `cz_k_support.py`, `v211_c.py`, `v211_w.py`, `v211_a.py`,
  `representative_prism_checkpoints.py`, `strict_oof_selection.py` and
  `cz_l256_nowcast.py` are byte-identical to that authority commit;
- `v211_joint_stability.py` is exactly the reviewed zero-C patched blob;
- `level_reconstruction.py` is exactly the reviewed reporting-only patch that
  names MSE-relative and RMSE-relative persistence skill separately;
- the task remains 2-second sampling, L256, H4/W1/W0=1, with a 260-point
  dependency purge and target `D[t+3]-D[t-1]`.

The authoritative final checkpoint currently emits the formal ladder from
`K+C` through `Joint`.  A pure-K formal checkpoint adapter is still
`NOT_YET_RUN`.  E2--E6 are also `NOT_YET_RUN` until their perturbation and
candidate-universe adapters rerun the full authority chain.  The corrected
runner must not substitute compact Ridge models for those experiments.

## Validated execution receipt

The valid corrected run is based on execution commit
`551c491db200d362c65b92c928d5a98f86bf96f4` and is stored privately at:

```text
/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_20260922_R3/
```

The run passed scope, environment, pilot, pilot acceptance, development,
development reconciliation, selection freeze, common-support freeze,
checkpoint sealing, formal test and report.  Development accessed neither
test nor OOD data.  The sealed manifest contains 32 checkpoint files and the
report privacy audit has no violations.  The server-side targeted regression
suite passed 54 tests.

The registered task is `CZ_DIAM_RAW2S_CURRENT_L256_H4`: 2-second sampling,
256-point strictly-past history, H/W/W0 = 4/1/1, hence an 8-second forecast.
Rod1 to Rod2 and Rod2 to Rod1 were fitted independently.  Formal test metrics
are:

| Direction | Stage | Delta RMSE | Delta R2 | Reconstructed level R2 | Persistence skill |
| --- | --- | ---: | ---: | ---: | ---: |
| Rod1 to Rod2 | K+C | 0.015238 | 0.093504 | 0.998294 | 0.094468 |
| Rod1 to Rod2 | K+C+Delta-W | 0.015238 | 0.093504 | 0.998294 | 0.094468 |
| Rod1 to Rod2 | K+C+A ablation | 0.013819 | 0.254551 | 0.998597 | 0.255344 |
| Rod1 to Rod2 | K+C+Delta-W+A | 0.013819 | 0.254551 | 0.998597 | 0.255344 |
| Rod1 to Rod2 | Joint KWA | 0.013712 | 0.266011 | 0.998618 | 0.266791 |
| Rod2 to Rod1 | K+C | 0.016605 | 0.077985 | 0.999161 | 0.077993 |
| Rod2 to Rod1 | K+C+Delta-W | 0.016605 | 0.077985 | 0.999161 | 0.077993 |
| Rod2 to Rod1 | K+C+A ablation | 0.015223 | 0.225043 | 0.999295 | 0.225051 |
| Rod2 to Rod1 | K+C+Delta-W+A | 0.015223 | 0.225043 | 0.999295 | 0.225051 |
| Rod2 to Rod1 | Joint KWA | 0.014822 | 0.265376 | 0.999331 | 0.265383 |

The near-one level R2 and the approximately 0.265 delta R2 measure different
quantities and must not be interchanged.  Level R2 is high partly because the
diameter level is strongly persistent; persistence skill and delta R2 are the
appropriate checks for improvement over that easy level baseline.

`RUN_STATUS.json` is intentionally `PARTIAL`: the authority-backed H4 ladder
is complete from K+C onward, but pure K and E2--E6 have not been rerun.  The
failed pre-correction R1/R2 attempts and `CZ_RAW2S_H4_CORRECTED_R2` are not
valid result sources.

The pre-registered implementation and execution plan for completing pure K
and E2--E6 is `docs/CZ_RAW2S_H4_AUTHORITY_E1_E6_RERUN_PLAN_20260922.md`, with
its machine-readable companion at
`configs/cz_raw2s_h4_authority_e1_e6_rerun_plan_20260922.json`.  Both remain
`PLANNED_NOT_STARTED`; their presence is not evidence that E2--E6 ran.
