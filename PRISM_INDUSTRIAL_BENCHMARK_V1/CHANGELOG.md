# Changelog

## PRISM v2.1.1 - Prospective NeuroBEM MIMO audit

- Adds a new public-dataset experiment; it does not alter or rerun any prior
  Metro-P60, SRU, v2.1.2, or Joint-stability evidence.
- Treats each of the 96 flights as a split entity and every published
  continuous processed segment as a stricter no-crossing history entity.
- Adds a four-motor, four-output causal MISO FIR K audit using squared motor
  speed as a registered thrust proxy and rigid-body generalized-force targets.
- Adds optional W aerodynamic-context prediction and mature residual-state A
  stages. Their outputs are predictive diagnostics, not causal labels for
  airflow, drag, wind, or vortex-ring state.
- Adds ERA block-Hankel/SVD realization from frozen K Markov parameters only;
  W/A residuals are excluded from the realization.
- Freezes all splits, candidates, gates, row support, order candidates and the
  high-speed diagnostic before processed numeric values or test targets are
  read. Official-test parent flights stay locked until development freeze.
- Raw/processed NeuroBEM archives and sample-level predictions are excluded
  from Git and return bundles.

## PRISM v2.1.1 - Post-experiment prospective sample-support maintenance

- No new theory version is introduced and no estimator, candidate family,
  threshold, split, or model-selection rule is changed.
- Fixes the prospective implementation in which C1 previously truncated all
  anchors by a head-global maximum registered K history.
- C1 now materializes the maximal head-legal anchor universe; K candidates fit
  on profile-native support and direct local comparisons share one scoring
  support.
- Multi-channel common support begins only after K channel selection, at C
  assembly. C recomputes the best active K comparator on that support.
- W, A, Joint, and final materialization inherit the frozen C assembly support.
- The completed Metro-P60 experiment was not rerun. Its evidence remains bound
  to generating commit `76231f3959c15183fbc781eb238034085ee71fc1` and remains
  valid under the historical head-level common-support protocol, not
  native-support optimal.
- No historical FREEZE/FINAL/result/prediction artifact and no test/OOD result
  was changed or reaccessed by this maintenance.

## PRISM v2.1.1 - Joint Predictive Stability Practice Update

- Refines executable estimator practice within canonical PRISM v2.1.1; no standalone v2.2 theory is retained.
- Separates numerical ridge from explicit predictive ridge and registers compressed/full representations of the same frozen K support.
- Makes no K/C/W/A estimator, data-split, Joint gate-threshold, or candidate-family change.
- Reuses M2--M4 and migrates the abf7 M5 development evidence without recomputation or loss changes.
- Retains PF-independent freeze semantics and registers Joint as an optional predictive enhancement.
- Freezes `J_SELECTED` vs `PF_SELECTED` as the primary final comparison before lockbox access.
- Test/OOD remained untouched before the canonical M6 freeze; no post-test reselection is permitted.

## PRISM v2.1.2 - PF Independent Freeze Semantics

- Clarifies practice-contract, freeze, assembly, and materialization semantics;
  there is no estimator, hyperparameter, candidate-family, or threshold change.
- Retains the v2.1.2 Joint OOF correction development results unchanged and
  does not recompute M2--M5.
- Makes Physical-First and Joint independently freeze-eligible evidence
  routes. Physical-First is the mandatory formal route; Joint remains an
  optional predictive enhancement.
- Excludes a development-unsupported Joint route before test/OOD while
  allowing a fully valid Physical-First route to freeze independently.
- Makes M6 candidate registration and later M7/M8 behavior route-aware, so a
  PF-only freeze cannot create, fit, predict, or report formal Joint models.
- Distinguishes different outcomes from the same gate implementation applied
  to different predictions from a true same-evaluation implementation
  inconsistency.
- This practice-contract clarification was derived and frozen before any new
  test/OOD access. `test_accessed=false` and `ood_accessed=false` remain hard
  development guards.

## PRISM v2.1.2 - Joint OOF Protocol Correction

- Preserves the PRISM v2.1.1 Metro-P60 failed-stop branch, release, and result
  namespace as immutable formal audit evidence.
- Does not modify the W degradation/identity mechanism. Identity W remains
  exactly equivalent to skipping W.
- Does not modify K, C, or A model semantics, families, supports, profiles, or
  selection rules.
- Does not modify the Joint candidate family, ridge/penalty grids, one-SE rule,
  practical activation gate, positive-fold requirement, input-path thresholds,
  coefficient threshold, or numerical certificate.
- Corrects only the Joint development fold support. v2.1.1 used prior OOF
  validation chunks as the next Joint fit pool; v2.1.2 uses each original
  registered inner train/evaluation pair.
- Restricts W `PHYSICAL_OOF.parquet` to cross-stage fold/provenance auditing;
  it is never a Joint training pool.
- Adds strict C/W/Joint fold-ID hashes, complete four-fold coverage, fold-local
  C latent/W basis construction, and train-only inner A target-state access.
- Adds a diagnostic failure class so an MSE-only preservation failure is
  reported as `INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED`, without
  changing PASS/FAIL semantics.
- Ignores the dedicated v2.1.2 result namespace so the M0 clean-worktree guard
  remains valid after the runner creates its output directory.
- Test and OOD data were not accessed before this correction and are not used
  to motivate or tune it.
# PRISM v2.1.1 NeuroBEM Multi-Horizon × Wiener-Prior Audit

- Registers a direct forecast grid of 1, 4, 8, 20, 40, and 80 samples instead of relying only on the prior one-step audit.
- Defines mature residual state by age at the prediction origin, so the actual target lag is `horizon + age`.
- Preserves canonical latent-only Wiener evidence as W1 and separately labels W2 as an aerodynamic-context extension; W0 remains exact identity.
- Freezes all W0/W1/W2 arms for formal evaluation without a post-hoc global arm winner.
- Declares the experiment a `POST_LOCKBOX_PROSPECTIVE_EXTENSION` and explicitly discloses the completed R1 official-test access.
- Freezes horizons, candidates, support, baselines, metrics, statistics, and plots before any new multi-horizon test score is computed.
- Results and retained failures will be appended only after the single formal extension test access.
