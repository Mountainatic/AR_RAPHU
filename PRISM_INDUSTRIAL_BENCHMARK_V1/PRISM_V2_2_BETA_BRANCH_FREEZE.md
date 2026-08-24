# PRISM v2.2 (beta) — isolated branch freeze

## Status

**Canonical version name:** `PRISM v2.2 (beta)`  
**Canonical branch:** `prism-v2-2-beta-ct`  
**Parent stable line:** PRISM v2.1.1, branch `prism-v2-1-1-metro-p60-joint-stability-final`  
**Development predecessor:** `prism-ct-v0-2-silicon-20260823`  
**Freeze intent:** experimental minor-version advance from v2.1.1; not a replacement for the v2.1.1 stable baseline.

> **ISOLATION / ROLLBACK RULE**
>
> This file and all PRISM v2.2(beta)-specific files apply **only** to the branch `prism-v2-2-beta-ct`.
> If PRISM is later rolled back to v2.1.1, reconstructed from the stable line, or reorganized without an explicit request to include v2.2(beta), **do not use this document, its CT-specific assumptions, its experimental results, or its branch-only files as authoritative PRISM definitions.**
> The canonical rollback reference remains the frozen PRISM v2.1.1 line.

## Why this is v2.2(beta), not PRISM-CT v0.2

`PRISM-CT v0.2` was the development codename used while testing continuous-time state representations. The retained changes are now treated as a coherent experimental minor-version extension of PRISM v2.1.1. Therefore the version is frozen as **PRISM v2.2 (beta)**.

The beta label is mandatory because the CT route has not yet completed the full public-dataset benchmark, full certificate integration, and publication-grade ablation suite required for a stable PRISM release.

## Inherited PRISM principle

PRISM v2.2(beta) inherits the v2.1.1 philosophy and does not redefine the stable baseline. In particular, the following principles remain mandatory:

- causal information contracts and availability control;
- stagewise construction rather than unrestricted end-to-end black-box fitting;
- explicit K/C/DeltaW/A semantics;
- train/validation selection separated from final holdout evaluation;
- numerical/support certificates recorded independently of predictive score;
- physics-first and accuracy-first roles remain distinguishable;
- no silent use of final holdout data for route, profile, penalty, or assembly selection.

## Frozen v2.2(beta) structural change

The temporal representation is extended from a primarily discrete/profile view to an explicit **delay-state-scale decomposition**:

`T/E -> {Discrete Delay || Stable CT-Multires || optional certified CT-Absolute} -> branch predictors -> constrained prediction-level late assembly Gamma_CT -> DeltaW -> residual-state A -> output`

The branch semantics are:

1. **Discrete Delay (D):** explicit causal lag information; interprets *when* an informative event occurred.
2. **Stable CT-Multires (M):** differences between adjacent stable continuous-time states; interprets *which timescale is changing*.
3. **Stable CT-Absolute (S):** slow continuous-time state levels; interprets *what slow dynamical state the system is in*. This branch is optional and must pass conditioning certification.
4. **Persistence anchor (P):** explicit zero-correction fallback available to the assembly layer.

The preferred hybridization point is the **prediction/assembly level**, denoted `Gamma_CT`, not unrestricted raw feature concatenation.

**Notation clarification:** `Gamma_CT` is not the v2.1.1 module `A`. In v2.1.1, `A` remains reserved for the mature second-stage residual-state module. Earlier development wording such as “A-level late assembly” meant only “assembly level” and is superseded by the unambiguous `Gamma_CT` notation.

## Frozen continuous-time basis

For the current silicon beta protocol, cadence is provisionally `dt = 2 s` and the fixed time constants are:

`tau = [10, 30, 60, 120, 300, 600, 1200, 2400, 4800, 7200] s`.

Each state obeys

`dz_r/dt = -(1/tau_r) z_r + (1/tau_r) x`

with exact zero-order-hold discretization

`z_r[k] = exp(-dt/tau_r) z_r[k-1] + (1-exp(-dt/tau_r)) x[k]`.

The v2.2(beta) multiresolution state is increment-only:

`[x-z_tau1, z_tau1-z_tau2, ..., z_tau(R-1)-z_tauR]`.

This choice is deliberate: direct stacking of absolute CT states produced severe multicollinearity in the silicon development audit, while adjacent-scale increments substantially reduced the condition number.

## Frozen assembly rule

The beta late assembly `Gamma_CT` combines separately fitted branch corrections through a nonnegative simplex with an explicit persistence anchor:

`Delta_y_hat = w_D Delta_D + w_M Delta_M + w_S Delta_S + w_P * 0`

subject to

`w_i >= 0`

and

`w_D + w_M + w_S + w_P = 1`.

The assembly weights must be learned without access to the final holdout.

This preserves branch semantics and allows explicit fallback when neither temporal branch is sufficiently supported.

## Numerical certificate status

The provisional standardized design condition-number hard-fail threshold for the CT development audit is:

`condition_number_hard_fail = 1e8`.

Observed silicon development behavior motivating the freeze:

- CT-Absolute: condition numbers roughly `1e9` to `2.7e9` -> rejected on those runs;
- CT-Multires: roughly `7.8e6` to `2.3e7` -> admissible on those runs;
- early raw Delay+CT concatenation can approach the hard boundary -> retained only as an ablation, not the preferred assembly mechanism.

These values are development evidence, not universal physical constants. A stable v2.2 release would require the threshold rule and sensitivity audit to be frozen across datasets.

## Support-certificate status

Input/state support checks remain recorded, but the current CT support check is **diagnostic-only** in v2.2(beta). It is not yet a universal hard kill switch because cross-sheet development experiments showed that coordinate-wise distribution shift can occur even when CT-Multires retains predictive transfer value.

A stable successor must define whether native support for CT states is measured in raw coordinate space, normalized state-energy space, or a certified dynamical manifold representation.

## What is explicitly NOT part of PRISM v2.2(beta)

The following are outside this freeze and must not be described as v2.2(beta) features unless a later version explicitly adds them:

- learned/free time constants `tau_r`;
- input-dependent poles or unrestricted `A_t`;
- Mamba/selective-SSM blocks in the physics-first trunk;
- unconstrained neural gating of branch weights;
- end-to-end black-box replacement of K/C/A;
- claiming CT latent states are measured physical quantities without independent physical validation;
- treating the two silicon sheets as confirmed independent boules unless their physical identity is separately verified.

## Interpretability position

The intended interpretation hierarchy is:

- Delay branch -> event timing / explicit delay;
- CT-Multires -> active dynamical timescale band;
- CT-Absolute -> slow dynamical state proxy;
- `Gamma_CT` weights -> auditable temporal-representation contribution/fallback fractions;
- residual-state `A` -> mature predictable residual state outside the frozen input pathway;
- certificates -> whether a representation is numerically/support admissible.

Thus v2.2(beta) aims to extend PRISM from historical-feature interpretability toward dynamical-state interpretability without turning the physics-first route into a generic Mamba-like black box.

## Evaluation status

Silicon experiments presently support the structural choice, especially for medium/long horizons and cross-sheet robustness, but remain **beta evidence**. They are not sufficient to supersede the frozen v2.1.1 benchmark tables.

Before a stable v2.2 release, at minimum the following are required:

- integration with the canonical PRISM stage contracts rather than standalone runners;
- frozen route/certificate protocol;
- public-dataset reruns against v2.1.1 under identical splits;
- full ablation: Delay only / CT only / late assembly / absolute CT / multires CT / persistence anchor;
- sensitivity audits for tau-bank limits, pole density, condition threshold, and assembly regularization;
- explicit interpretability and failure-mode audit.

## Rollback authority

If this beta line is abandoned, the correct rollback action is conceptually:

`PRISM v2.2(beta) isolated branch -> ignore -> resume from frozen PRISM v2.1.1`

Do **not** reverse-engineer v2.1.1 from this document. Do **not** use this document to resolve ambiguity in the stable PRISM definition. Use the v2.1.1 frozen theory/code/contracts directly.