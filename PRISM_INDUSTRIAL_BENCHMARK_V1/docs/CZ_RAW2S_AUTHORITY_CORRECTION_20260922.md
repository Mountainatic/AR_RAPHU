# CZ raw-2s H4 authority correction

Status: `IMPLEMENTED_NOT_YET_RERUN`

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
- the task remains 2-second sampling, L256, H4/W1/W0=1, with a 260-point
  dependency purge and target `D[t+3]-D[t-1]`.

The authoritative final checkpoint currently emits the formal ladder from
`K+C` through `Joint`.  A pure-K formal checkpoint adapter is still
`NOT_YET_RUN`.  E2--E6 are also `NOT_YET_RUN` until their perturbation and
candidate-universe adapters rerun the full authority chain.  The corrected
runner must not substitute compact Ridge models for those experiments.

