# PRISM v2.0 Changelog from v1.3

## 1. Version decision

PRISM v1.3 is retained as the authoritative theory for:

- channel-specific multirate profiles;
- Urysohn-first Physics-First route;
- mature residual AR;
- K-Joint AR predictive route;
- identifiability, Schur/Gram and deployment boundaries.

PRISM v2.0 does not invalidate v1.3. It embeds v1.3 as one assembly:

\[
\text{v1.3 Physics-First}
\equiv
E_{\mathrm{channel}}
+K
+C_{\mathrm{add}}
+W_{\mathrm{identity}}
+A_{\mathrm{mature}}.
\]

## 2. Core conceptual change

### v1.3

A two-route theory centered on:

1. Urysohn-first K then mature residual AR;
2. K-Joint AR for total prediction.

### v2.0

A modular operator family centered on:

1. mandatory causal/time contract;
2. optional process-variable response;
3. optional joint channel basis;
4. optional Wiener readout;
5. optional state or mature residual memory;
6. separate predictive joint route;
7. explicit neutral elements and assembly cards.

## 3. New modules

| Module | v1.3 status | v2.0 status |
|---|---|---|
| E scale encoder | implicit in channel profile | explicit interface |
| K input response | core | retained |
| C channel fusion | mostly additive compressed outputs | additive compressed, joint basis, sparse interaction |
| W Wiener readout | absent | optional, identity neutral element |
| A state/residual | mature residual AR | generalized state-only or mature residual module |
| J joint predictor | K-Joint AR | generalized predictive joint route |
| R regime selector | absent | interface only, deferred benchmark |
| online adaptation | deployment topic | interface only, frozen benchmark |

## 4. Two formal faces

v2.0 formally permits:

### State/system-identification face

\[
K=0,\qquad \widehat z=A(Y^-).
\]

It predicts state evolution but does not claim input-response identification.

### Process-variable/soft-sensor face

\[
\widehat z=W(C(K(U^-,X^-)))
\]

or

\[
\widehat z=W(C(K(U^-,X^-)))+A(R_{\mathrm{mature}}^-).
\]

## 5. New ownership constraints

1. K owns within-channel dynamic response;
2. C owns cross-channel fusion;
3. W owns only residual static curvature orthogonal to the K/C linear physical space;
4. A owns only mature residual state, optionally residualized against the frozen physical space;
5. J owns total prediction only.

## 6. New neutral elements

- K exact-zero;
- C additive;
- W identity;
- A exact-zero;
- R single-regime;
- adaptation frozen.

These make module removal a formal model result rather than an implementation failure.

## 7. New training sequence

v1.3:

\[
K\rightarrow\text{freeze}\rightarrow A_{\mathrm{res}}.
\]

v2.0:

\[
E/K\rightarrow C\rightarrow\text{freeze}
\rightarrow W_{\perp}\rightarrow\text{freeze}
\rightarrow A_{\mathrm{mature},\perp}\rightarrow\text{freeze}.
\]

Joint predictive training remains separate.

## 8. New experiment semantics

Because v2.0 was proposed after inspection of C6 V2 primary-head results:

- re-evaluation on the same primary test/OOD heads is post-hoc exploratory;
- previously unaccessed registered horizons, availability views and proxy views are used for prospective internal confirmation;
- old C6 V2 baseline predictions are reused rather than recomputed;
- no complete GPU benchmark is required before the modular CPU structure is tested.

## 9. Empirical motivation, not theory content

The completed C6 V2 benchmark showed that fixed model families have process-dependent strengths. This motivated modularity, but no numerical ranking is imported into the theory document. The new benchmark must expose every task to the same module grid and may select identity/exact-zero states.

## 10. Archived interpretations

The following interpretations are explicitly rejected:

- adding Wiener only to datasets where the old test score was poor;
- treating state-only PRISM as input physics identification;
- treating orthogonality as causal independence;
- using a nonlinear readout to re-explain K's linear contribution;
- using residual AR to read raw inputs;
- claiming full pre-registration for primary-head v2 test results.


## 11. Numerical freeze

The first v2 draft left implementation-facing values in `PROPOSED_FREEZE`. On 2026-08-04 they were replaced by `PRISM_V2_MODULAR_ASSEMBLY_NUMERICAL_FREEZE_V1`. The freeze fixes the one-SE formula, practical activation gates, fold-direction stability, row caps, K/C/W/A/J grids, joint-basis dimensions, interaction screening, spline knots and direction rules, orthogonality tolerances, solver rescue rules, bootstrap support criteria, OOD degradation thresholds, stopping conditions and parameter-count semantics. No unresolved numerical field remains; code must stop instead of inventing a default.
