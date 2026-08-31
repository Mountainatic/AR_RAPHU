# SRU v2.2 full-KWA adapter regression audit — run #5

Status: **NON-CANONICAL / DIAGNOSTIC ONLY**

GitHub Actions run: `33364116552` (run #5)

This run successfully executed the adapter code, but subsequent inheritance audit found that the adapter reintroduced implementation semantics already prohibited and repaired in the frozen PRISM v2.1.1 SRU correction. Therefore its numerical output must not be used as canonical evidence for PRISM v2.2.

## Observed diagnostic result

- K_GAMMA: RMSE 0.018864074, R2 0.893496409
- K_GAMMA_W: RMSE 0.018864074, R2 0.893496409
- K_GAMMA_W_A: RMSE 0.013539937, R2 0.945131061
- persistence: RMSE 0.018864074, R2 0.893496409
- Gamma persistence weight: approximately 1
- W: IDENTITY_CORRECTION
- A: MATURE_RESIDUAL_AR, profile (1,4)

## Regression 1 — C erased an already-active K path

The adapter selected C ridge alpha by ordinary one-SE with complexity key `-alpha`. A very large ridge value could therefore be preferred merely because it was treated as simpler. This shrank the C output to near zero and then caused `C_INPUT_PATH_NOT_PRESERVED` for all D/M/S branches.

This is the same failure class already documented in `PRISM_V2_1_1_SRU_REPAIR_PLAN.md`: ridge is numerical stabilization, not structural complexity; C may not silently erase an active K path. The frozen v2.1.1 SRU correction requires smallest-stable ridge and a fallback to `BEST_ACTIVE_K_CHANNEL` when a C representation cannot preserve the input path.

## Regression 2 — W was evaluated after the C-induced latent collapse

Because all D/M/S branches were rejected, Gamma_CT contracted to the zero-delta persistence corner. The W latent `gamma_oof` consequently had no numerically resolvable variation. Nonlinear W candidates could not meaningfully participate; the selected identity correction therefore cannot be interpreted as evidence that W is unnecessary on SRU.

## Repair rule

The next run restores the already-frozen v2.1.1 SRU C inheritance contract without changing K/W/A candidate universes or activation thresholds:

1. C ridge: smallest numerically stable value;
2. C input-path preservation threshold: max(1e-8 target-variance ratio, 0.10 times best-active-K variance ratio);
3. C MSE no worse than 1.02 times best-active-K MSE;
4. at least one non-intercept coefficient magnitude >= 1e-10;
5. failure of compressed C falls back to BEST_ACTIVE_K_CHANNEL instead of deleting the branch.

Only after this restoration may W activation/non-activation be interpreted scientifically.
