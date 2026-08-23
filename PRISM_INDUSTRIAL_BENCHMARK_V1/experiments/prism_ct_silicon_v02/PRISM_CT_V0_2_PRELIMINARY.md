# PRISM-CT v0.2 — Silicon preliminary note

## Scope

This branch is an experimental extension of the frozen PRISM v2.1.1 baseline. It does **not** replace the frozen implementation and should not be merged into the benchmark baseline until the CT protocol is frozen and rerun.

The v0.2 purpose is to test a structural decomposition of temporal information:

1. **Discrete delay branch** — explicit lagged states, answering *when did an informative change occur?*
2. **Stable CT absolute-state branch** — a bank of first-order continuous-time modes, answering *what slow state is the process currently in?*
3. **Stable CT multiresolution branch** — adjacent-scale state increments, answering *which timescale is currently changing?*
4. **A/routing audit** — conditioning and input-support certificates decide which branches are numerically admissible before predictive selection.

## Frozen exploratory CT basis

Cadence is provisionally fixed at 2 s. Time constants are:

`10, 30, 60, 120, 300, 600, 1200, 2400, 4800, 7200 s`

Each stable state satisfies

`dz_r/dt = -(1/tau_r) z_r + (1/tau_r) x`

and is discretized exactly under zero-order hold:

`z_r[k] = exp(-dt/tau_r) z_r[k-1] + (1-exp(-dt/tau_r)) x[k]`.

The multiresolution representation used by v0.2 is **increment-only**:

`[x-z_tau1, z_tau1-z_tau2, ..., z_tau(R-1)-z_tauR]`.

This intentionally removes the repeated instantaneous state from the CT branch.

## Conditioning result

On the provided silicon sheets, directly stacking the absolute CT states produces standardized design condition numbers around `1e9` to `2.7e9`, above the provisional `1e8` hard-fail threshold.

After converting the CT branch to scale increments, the comparable condition numbers fall to roughly `7.8e6` to `2.3e7` in the train-only audit used during development. Therefore:

- `CT-Absolute`: rejected by the v0.2 conditioning certificate on these runs.
- `CT-Multires`: numerically admissible.
- `Delay + CT-Multires` early feature concatenation: can again approach the hard threshold and is retained only as an auditable ablation, not an automatically trusted route.

This is the first concrete reason to prefer **scale-increment CT states** over a naive bank of EMA-like absolute states.

## Route-then-holdout protocol

The v0.2 runner uses:

- horizons: `1, 5, 15, 30, 60, 120, 300, 600` steps;
- four expanding train/source-only inner folds;
- 600 s selection separation;
- one-standard-error selection with a complexity tie-break;
- outer holdout never participates in route selection;
- descriptive branch-by-branch test ablations are explicitly marked `NOT_FOR_SELECTION`.

Current diameter `y_t` is causally available and is included in the temporal state bank. The label remains future diameter `y[t+h]`.

## Preliminary observations

### 1. CT multiresolution contains substantial held-out signal

When each numerically admissible branch is inspected descriptively on the final holdout, Sheet1 is consistently best served by `CT-Multires` in this run. Its persistence-MSE skill rises from about 4.3% at one step to about 45.2% at 2 min, 53.1% at 4 min and 56.9% at 10 min.

These numbers are **descriptive-only** because the final holdout is not allowed to choose the route.

### 2. Sheet2 benefits from combining delay and CT information, but early fusion is near the conditioning boundary

On the descriptive Sheet2 holdout, the `Delay + CT-Multires` concatenation is strongest and reaches about 41.2% persistence-MSE skill at 10 min and 68.4% at 20 min. However its standardized condition number is close to `1e8` on this split.

This argues against simply concatenating all temporal features. The next assembly should combine separately fitted branch predictions at the PRISM A level rather than constructing one very wide feature matrix.

### 3. Static route selection is regime-sensitive

Train-only inner-fold routing is deliberately conservative and often selects persistence or delay even when another branch later performs better on the final holdout. The intermediate 60–80% validation segment can also disagree with the final 80–100% segment.

This is not evidence that the holdout should be used for tuning. It instead exposes a real structural problem: **the preferred temporal representation changes with operating regime**.

For PRISM-CT this points toward a constrained, state-dependent assembly/router rather than one globally fixed branch per horizon.

### 4. Cross-sheet transfer is asymmetric

Under source-only route selection, one direction selects delay at some medium horizons and obtains positive transfer skill; the reverse direction is more conservative. Input-support checks are recorded independently of target error.

Before interpreting this as cross-boule evidence, the physical identity of Sheet1 and Sheet2 must be confirmed.

## Architectural consequence

The v0.2 evidence favors the following next implementation shape:

`T/E -> {Delay branch || Stable CT-Multires branch || optional certified CT-Absolute branch} -> K/C branch predictors -> A-level constrained assembly -> DeltaW -> output`

The key change from the first CT prototype is that **hybridization should occur at prediction/assembly level, not by raw feature concatenation**.

A future selective gate may be state dependent, but its inputs and admissible weights should remain constrained so that the stable CT modes and branch certificates remain interpretable.

## Status

`PRISM-CT v0.2` is an exploratory branch. Current results establish useful structural evidence but are not yet a paper benchmark. The next step is to implement late branch assembly inside the PRISM stagewise routing framework and rerun under a fully frozen silicon protocol.
