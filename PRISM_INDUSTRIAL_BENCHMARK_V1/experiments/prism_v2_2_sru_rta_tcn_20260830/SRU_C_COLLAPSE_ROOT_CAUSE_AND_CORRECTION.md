# PRISM v2.2 SRU C-collapse root cause and correction

## Status

This note records an implementation-contract defect found after the first full SRU v2.2 K/W/A run and the correction used for the formal rerun. It is an audit record, not a post-hoc score-selection rule.

## 1. Superseded V1 behavior

Protocol `PRISM_V2_2_BETA_SRU_RTA_TCN_FULL_KWA_MATCHED_V1` used the v2.2 SRU adapter's C rule:

1. score C ridge alphas on inner expanding folds;
2. apply ordinary one-SE selection;
3. use complexity key `(-alpha,)`, so the largest alpha inside the one-SE set is preferred;
4. refit C;
5. reject the entire D/M/S branch if the refitted C output fails the input-path variance/coefficient gate.

On SRU H2S, the mean-validation optimum for D/M/S was alpha `151.99110829529332`, but one-SE admitted much larger penalties and selected `1e8`. The resulting C prediction variance was driven to approximately numerical-zero relative to the delta-target variance. All three branches were then rejected as `C_INPUT_PATH_NOT_PRESERVED`.

Consequences:

- Gamma_CT contracted to the persistence / zero-delta corner;
- W received a degenerate zero-delta latent and selected `IDENTITY_CORRECTION`;
- A was left to explain nearly all remaining delta-target dynamics.

The resulting V1 nested routes were:

- K_GAMMA: R2 `0.893496409` (identical to persistence)
- K_GAMMA_W: R2 `0.893496409`
- K_GAMMA_W_A: R2 `0.945131061`
- PERSISTENCE: R2 `0.893496409`

This result remains retained for audit but is superseded for formal reporting.

## 2. Evidence that K was not actually inactive

Strict K numerical admission had already retained active channels before C erased them:

- D active channels: `[0, 2]`
- M active channels: `[0, 2]`
- S active channels: `[2]`

The strongest channel-2 K candidates improved fold MSE over exact-zero by roughly 21--26% with improvement on all four selection folds. Therefore branch rejection after C shrinkage was not equivalent to evidence that the exogenous K path was absent.

## 3. Best-mean C counterfactual: diagnostic only

A diagnostic changed only C ridge selection from largest-alpha one-SE to minimum mean OOF validation MSE. It retained strict K and all downstream W/A rules.

Results:

- D: alpha `151.99110829529332`, variance ratio `0.3222226734`, path pass
- M: alpha `151.99110829529332`, variance ratio `0.3142201977`, path pass
- S: alpha `151.99110829529332`, variance ratio `0.2466174029`, path pass
- Gamma weights: D `0.02370065`, M `0.02363524`, S `0.01990974`, persistence `0.93275437`
- W: `MONOTONE_I_SPLINE_CORRECTION`, active
- A: `MATURE_RESIDUAL_AR`, active
- K_GAMMA R2: `0.896847843`
- K_GAMMA_W R2: `0.917336484`
- K_GAMMA_W_A R2: `0.938643597`

This establishes the causal diagnosis: once C no longer annihilates the K-derived latent, W becomes active and provides substantial incremental prediction gain.

The best-mean C rule is NOT adopted for formal reporting because it was introduced as a post-diagnostic counterfactual after V1 results were observed.

## 4. Best-mean A counterfactual: diagnostic only

With strict K, best-mean C and the current W selector, allowing A to choose its minimum-mean OOF candidate (subject to the existing activation guard) produced:

- W: `MONOTONE_I_SPLINE_CORRECTION`, active
- A: `(MATURE_RESIDUAL_AR, (1,16), 0.18738174228603832, 30.0)`
- K_GAMMA R2: `0.896847843`
- K_GAMMA_W R2: `0.917336484`
- K_GAMMA_W_A R2: `0.953798646`

This is also diagnostic-only and must not replace the inherited formal A selector.

## 5. Contract restoration used for formal rerun

The formal correction restores the already-frozen v2.1.1 SRU C semantics inside the v2.2 temporal adapter:

1. C ridge is numerical stabilization, not structural simplicity;
2. select the smallest numerically stable ridge alpha;
3. evaluate input-path preservation on inner-fold OOF predictions;
4. required C variance is the larger of:
   - `1e-8` of target variance, and
   - `0.10` times the best-active-K variance ratio;
5. C MSE must be no worse than `1.02` times best-active-K MSE;
6. at least one non-intercept coefficient must have absolute magnitude >= `1e-10`;
7. if the C fusion fails this gate while K is active, fall back to `BEST_ACTIVE_K_CHANNEL` rather than deleting the exogenous branch.

The correction is frozen as protocol:

`PRISM_V2_2_BETA_SRU_RTA_TCN_FULL_KWA_MATCHED_V2_C_RESTORED`

with config:

`configs/prism_v22_sru_rta_tcn_full_c_restored.json`

and runner:

`scripts/run_prism_v22_sru_full_restored_c.py`

## 6. Pre-formal restoration audit

The contract-restoration audit (before the dedicated formal matched rerun) produced:

- D: active K `[0,2]`, C fallback to `BEST_ACTIVE_K_CHANNEL`
- M: active K `[0,2]`, C fallback to `BEST_ACTIVE_K_CHANNEL`
- S: active K `[2]`, `ADDITIVE_COMPRESSED` C passes with alpha `1e-8`
- Gamma persistence weight: `0.9286551948`
- W: `NATURAL_CUBIC_CORRECTION`, active
- A: `MATURE_RESIDUAL_AR`, active

Nested routes:

- K_GAMMA: RMSE `0.018562782`, R2 `0.896871333`
- K_GAMMA_W: RMSE `0.016936661`, R2 `0.914148306`
- K_GAMMA_W_A: RMSE `0.014768968`, R2 `0.934717980`
- PERSISTENCE: RMSE `0.018864074`, R2 `0.893496409`

Using squared RMSE to decompose the improvement from persistence to KWA, approximately:

- K/Gamma contribution: 8.2% of total MSE reduction
- W contribution: 41.9%
- A contribution: 49.9%

Thus the exogenous K+W route contributes about 50.1% of total MSE improvement and the mature residual state A contributes about 49.9%. This is materially different from the superseded V1 decomposition in which C collapse made A appear to supply essentially all non-persistence improvement.

## 7. Reporting rule

For the paper / formal benchmark:

- retain V1 as an implementation-bug audit only;
- retain best-mean C/A runs as diagnostics only;
- report the V2 C-restored formal rerun as the corrected full-v2.2 SRU result;
- continue to label original RTA-TCN as input-only and full KWA as a richer dynamic-record-time information set;
- do not claim a strictly information-matched PRISM-KWA vs original RTA-TCN comparison;
- report nested K/Gamma, +W and +A metrics so the exogenous vs mature-residual contribution is visible rather than hidden by the final score.
