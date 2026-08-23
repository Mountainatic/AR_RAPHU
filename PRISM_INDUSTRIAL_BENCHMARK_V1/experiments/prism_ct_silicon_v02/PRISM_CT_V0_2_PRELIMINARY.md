# PRISM-CT v0.2 — Silicon preliminary note

## Scope

This branch is an experimental extension of the frozen PRISM v2.1.1 baseline. It does **not** replace the frozen implementation and should not be merged into the benchmark baseline until the CT protocol is frozen and rerun.

The v0.2 purpose is to test a structural decomposition of temporal information:

1. **Discrete delay branch** — explicit lagged states, answering *when did an informative change occur?*
2. **Stable CT absolute-state branch** — a bank of first-order continuous-time modes, answering *what slow state is the process currently in?*
3. **Stable CT multiresolution branch** — adjacent-scale state increments, answering *which timescale is currently changing?*
4. **A-level constrained late assembly** — separately fitted dynamic branches are combined with nonnegative weights and an explicit persistence anchor.
5. **Numerical/support audit** — conditioning is a hard admissibility check; cross-domain input support is recorded diagnostically in v0.2 but is not yet a frozen hard gate.

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

After converting the CT branch to scale increments, comparable train-only condition numbers fall to roughly `7.8e6` to `2.3e7`. Therefore:

- `CT-Absolute`: rejected by the v0.2 conditioning certificate on these runs.
- `CT-Multires`: numerically admissible.
- `Delay + CT-Multires` early feature concatenation: can again approach the hard threshold and remains descriptive-only.

This is the first concrete reason to prefer **scale-increment CT states** over a naive bank of EMA-like absolute states.

## Static route-then-holdout protocol

The first v0.2 runner uses:

- horizons: `1, 5, 15, 30, 60, 120, 300, 600` steps;
- four expanding train/source-only inner folds;
- 600 s selection separation;
- one-standard-error selection with a complexity tie-break;
- outer holdout never participates in route selection;
- descriptive branch-by-branch test ablations explicitly marked `NOT_FOR_SELECTION`.

Current diameter `y_t` is causally available and is included in the temporal state bank. The label remains future diameter `y[t+h]`.

## Why late assembly was added

Static global route selection is regime-sensitive. Train-only folds, the 60–80% validation segment, and the final 80–100% holdout can prefer different temporal representations. This is not a reason to tune on the holdout; it is evidence that one globally fixed temporal branch is too restrictive.

Raw early concatenation is also unattractive because it can recreate the conditioning problem. v0.2 therefore adds prediction-level assembly rather than wider feature matrices.

For admissible dynamic branch predictions `d_j`, the A-level assembly fits

`delta_y_hat = sum_j w_j d_j`

with an explicit zero-delta persistence branch and constraints

`w_j >= 0`, `w_persistence >= 0`, `sum_j w_j + w_persistence = 1`.

The weights are fitted only on the 60–80% validation/source segment. The test/destination target is never used for the weights. This gives the model a transparent shrinkage path back to persistence instead of allowing unstable negative cancellation between branches.

## Preliminary late-assembly results

### Within Sheet1

Persistence-MSE skill of the validation-fitted late assembly on the final holdout:

| horizon | skill | R² |
|---:|---:|---:|
| 2 s | ~0.0% | ~1.000 |
| 10 s | 0.4% | 0.999 |
| 30 s | 3.7% | 0.995 |
| 1 min | 7.7% | 0.987 |
| 2 min | 15.5% | 0.966 |
| 4 min | 22.6% | 0.899 |
| 10 min | 33.8% | 0.477 |
| 20 min | 49.0% | -0.536 |

The 20 min R² remains poor because this tail segment is intrinsically difficult, but MSE is still substantially lower than persistence.

### Within Sheet2

| horizon | skill | R² |
|---:|---:|---:|
| 2 s | ~0.0% | 0.997 |
| 10 s | 0.1% | 0.979 |
| 30 s | 0.8% | 0.912 |
| 1 min | 1.6% | 0.813 |
| 2 min | 4.5% | 0.768 |
| 4 min | 7.2% | 0.587 |
| 10 min | 12.5% | -0.310 |
| 20 min | 36.3% | -0.738 |

Again, long-horizon R² warns against presenting relative skill alone.

### Cross Sheet1 -> Sheet2

The source-validation-fitted late assembly obtains nonnegative persistence skill at every tested horizon:

`0.0%, 0.1%, 1.0%, 2.2%, 7.1%, 14.6%, 27.8%, 36.3%`

for `2 s, 10 s, 30 s, 1 min, 2 min, 4 min, 10 min, 20 min` respectively.

### Cross Sheet2 -> Sheet1

The corresponding skills are:

`0.0%, 0.1%, 1.6%, 4.0%, 13.8%, 22.1%, 25.0%, 42.4%`.

This is substantially more stable than choosing one global source branch per horizon.

## Important support-audit observation

The current global support rule uses a strict all-features 5-sigma criterion. It frequently marks the CT-Multires branch as out of support across sheets even when the target-free transferred predictor improves substantially over persistence.

Therefore v0.2 **records** support certificates but does not yet use this particular support score as a hard cross-domain kill switch. Freezing a hard support rule now would be premature. The next audit should distinguish harmless scale/energy shifts from true extrapolation, ideally using branch-level or state-energy support rather than requiring every CT increment coordinate to remain inside one global box.

## Architectural consequence

The current evidence favors:

`T/E -> {Delay predictor || Stable CT-Multires predictor || optional certified CT-Absolute predictor} -> constrained A-level assembly -> DeltaW -> output`

rather than

`[Delay features ; CT features] -> one wide regressor`.

The key change from the first CT prototype is that **hybridization now occurs at prediction/assembly level**. This preserves branch semantics, avoids raw-feature collinearity, exposes interpretable weights, and provides an explicit persistence fallback.

A future state-dependent gate can replace constant assembly weights, but it should remain constrained so that the stable CT modes and numerical certificates stay interpretable.

## Status

`PRISM-CT v0.2` now contains:

- stable CT pole-bank features;
- delay, absolute-state and multiresolution representations;
- conditioning and support audits;
- static route-then-holdout runner;
- constrained A-level late assembly;
- silicon within-sheet and cross-sheet runners;
- unit tests for stability, causality, rank deficiency and simplex assembly.

Local isolated tests for the new CT modules pass (`8 passed`).

This is still exploratory evidence, not a paper benchmark. The next implementation step is to connect the late assembly to the existing C4/A contract format and then freeze the silicon protocol before any further performance-driven changes.
