# WITHDRAWN AS CANONICAL THEORY

> **HISTORICAL EXECUTION DRAFT ONLY. CONTENT MERGED INTO PRISM v2.1.1 PRACTICE SECTION 11A.11.**
> `v2.2` is retained only as the historical label of the abf7 development execution and is not a canonical PRISM model/theory version.

# Historical draft: Joint Predictive Stability Extension

> Status: `THEORY_ONLY / ESTIMATOR_EXTENSION`  
> Parent: amended PRISM v2.1.1 implementation-safe stagewise routed modular assembly theory.  
> Scope: Joint predictive stability only. The K/C/W/A estimators, data boundaries, Physical-First route, input-path gate and freeze semantics are inherited without modification.

## 1. Evidence hierarchy

Physical-First remains an independently valid, independently freeze-eligible formal route. Joint remains an optional predictive enhancement. A Joint development failure may yield `PF_ONLY_FROZEN`; it does not invalidate a PF route that passed its own contracts. Joint may enter test/OOD only after its own development protocol, numerical, candidate-binding and predictive gates pass.

## 2. Frozen raw support and two Joint K representations

Let \(\mathcal S_K\) be the active channel support frozen by K. No Joint candidate may reselect raw channels or revive a K exact-zero channel. For each \(j\in\mathcal S_K\), the frozen profile, family, support and basis construction are retained.

The full representation is

\[
\Phi_K^F=[\Phi_{K,1},\ldots,\Phi_{K,m}],
\]

where each active channel contributes all of its frozen basis columns. The compressed representation is

\[
\Phi_K^C=[p_{K,1},p_{K,2},\ldots,p_{K,m}],
\]

where \(p_{K,j}\) is the channel-level prediction produced by the same frozen K structure. Thus \(\dim(\Phi_K^C)=m\), while normally \(\dim(\Phi_K^F)\gg m\).

`CHANNEL_COMPRESSED` is not the final C scalar. It retains one column for every active K channel. `CHANNEL_COMPRESSED` and `FULL_BASIS` therefore use the same raw information set and differ only in estimator degrees of freedom. Neither representation changes K selection.

## 3. Numerical ridge and predictive ridge

Every Joint design column is standardized using statistics fitted only on the current fit fold. Two penalties have distinct registered roles:

- \(\lambda_{num}\) exists only to obtain a valid numerical solve certificate;
- \(\eta_{pred}\ge0\) is a predictive shrinkage hyperparameter.

The predictive objective is

\[
\frac1n\lVert y-X\theta\rVert_2^2+\eta_{pred}\lVert\theta\rVert_2^2.
\]

For the sum-of-squares solver, the penalty matrix is

\[
P=P_{num}+n\eta_{pred}I.
\]

The numerical grid is scanned from small to large and the smallest value passing the registered certificate on every inner fold is frozen. Validation loss cannot choose \(\lambda_{num}\). Predictive ridge cannot conceal a numerically invalid bare design. The exact neutral boundary is \(\eta_{pred}=0\).

The v2.2 predictive grid is frozen before development access:

\[
\eta_{pred}\in\{0,10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1\}.
\]

Predictive block ratios equal one after standardization. The former block-ratio estimator is retained only as a non-selecting v2.1.2 legacy anchor.

## 4. Candidate family and development selection

The formal Joint route family remains exactly

\[
\{J_K,J_{KW},J_{KA},J_{KWA}\}.
\]

AR-only, K-zero, both-zero and any new route are excluded. A v2.2 candidate identity binds route, K representation, numerical alpha and predictive eta.

For each route and representation, \(\lambda_{num}\) is frozen by numerical certificates alone. The predictive eta is then selected by four-fold original registered \(T_i\rightarrow V_i\) OOF risk, using one-SE and preferring larger eta inside the one-SE set.

For each route, `CHANNEL_COMPRESSED` is the lower-complexity neutral representation. `FULL_BASIS` may replace it only when the registered one-SE condition, practical relative improvement and positive-fold fraction all pass. Otherwise the representation contracts to `CHANNEL_COMPRESSED`.

After this local selection, the existing guarded one-SE route selector compares \(J_K,J_{KW},J_{KA},J_{KWA}\), with \(J_K\) as neutral. Existing practical and fold-consistency thresholds are unchanged.

## 5. Gate, diagnostics and freeze

The selected Joint candidate is evaluated by the inherited input-path preservation gate without any threshold change. The gate retains its variance, coefficient, numerical-certificate and MSE-preservation checks against best active K.

The extension records the complete eta paths, representation comparisons, coefficient norms, effective degrees of freedom, prediction variances, worst fold, median fold and worst-to-median ratio. Diagnostic labels distinguish support for high-dimensional-basis instability, predictive-shrinkage deficit, a need for both controls, or failure of the registered controls. These are development model-selection diagnoses, not physical causal conclusions.

If the Joint gate passes, Joint may be added to the formal freeze. If it fails, Joint remains development-diagnostic-only and PF may still freeze independently. No unsupported Joint candidate may be evaluated on test/OOD.

## 6. Invariants

This extension does not modify K/C/W/A estimators, the data split, row caps, inner folds, W or A construction, Joint W joint fitting, candidate-binding rules, input-path thresholds, test/OOD guards, or PF-independent freeze semantics. It makes no physical causal claim about which K representation or ridge value is selected.
