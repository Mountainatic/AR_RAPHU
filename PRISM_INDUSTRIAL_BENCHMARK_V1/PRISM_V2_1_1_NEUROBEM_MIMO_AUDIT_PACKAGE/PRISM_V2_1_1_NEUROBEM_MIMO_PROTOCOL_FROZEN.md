# PRISM v2.1.1 NeuroBEM MIMO audit — frozen prospective protocol

Status: `FROZEN_BEFORE_PROCESSED_DATA_VALUE_ACCESS_AND_MODEL_IMPLEMENTATION`.

This is a new public-dataset experiment under the amended PRISM v2.1.1
practice contract. It is not a reinterpretation or rerun of Metro-P60. The
machine-readable authority is `PRISM_V2_1_1_NEUROBEM_MIMO_CONFIG_FROZEN.json`.

## Scientific scope

The experiment asks whether four causal motor-speed histories contain stable
linear dynamic evidence for the roll, pitch, yaw and body-z generalized-force
axes; whether a frozen-K residual admits a stable nonlinear correction from
measured body-motion context; whether a strictly mature residual state adds
predictive value; and whether the frozen linear Markov parameters support a
lower-order cross-axis MIMO realization.

The four motor inputs are deterministically encoded as centered squared angular
speed. K is therefore linear in a registered thrust proxy, not linear in raw
RPM. The targets are three rigid-body generalized torques and body-z generalized
force, computed from the published mass, diagonal inertia, body angular
velocity, angular acceleration and body-z acceleration.

W is a predictive aerodynamic-context correction. Body velocity is only a proxy
for relative airflow because wind is not observed. A is a mature predictable
residual state. Neither W nor A may be reported as unique causal identification
of drag, wind, vortex-ring state or another named aerodynamic mechanism.

## Entity and information isolation

The paper reports 96 flights, while the distributed `Flights.txt` and processed
archive contain the same 95 unique parent flight IDs. Those 95 observable
parents are the executable universe; no missing 96th entity is fabricated.
Every distributed flight is a parent entity and split group. Every processed
continuous segment is a stricter history entity because the publisher splits
flights at non-flying regions and Vicon dropouts. No lag, residual state,
standardization statistic, fold or sample identity may cross a segment.
Segments from one parent flight may never be divided among train, validation and
test.

All segments belonging to a parent flight named by the official test list are
locked test. Of the remaining parent flights, exactly 19 are validation groups,
chosen by the pre-registered SHA256 ordering; the remainder are train. Inner
selection uses four deterministic flight-grouped folds. The target at row `t`
uses information only through `t-1`.

## Stage order and lockbox

1. N0: source/hash/license/schema/run-boundary audit.
2. N1: immutable flight/segment registry, split manifest and sample identities.
3. N2: K development on the four original grouped folds.
4. N3: W development from frozen K OOF predictions.
5. N4: A development and PF route assembly.
6. N5: block-Hankel/ERA development from K Markov parameters only.
7. N6: development freeze with `test_accessed=false` and `ood_accessed=false`.
8. N7: one locked test/high-speed-subset access if and only if N6 permits it.
9. N8: paired segment-cluster bootstrap and final report without refitting.

N2--N5 must not read any test-segment numeric values. Source filenames and the
publisher's test list are metadata and may be read before N6.

## K, W, A and realization

K is a causal four-input FIR with histories 4, 8, 12, 20, 32 and 64 samples.
Each history fits on native support and all histories score on the 64-sample
local common support. Ridge is numerical only and is selected as the smallest
value passing the registered certificates, never by validation loss. Profile
selection uses guarded one-SE, the 2% regret guard, at least 1% improvement over
exact-zero and positive improvement in at least three of four folds.

W compares identity with fixed signed-quadratic and train-knot natural-cubic
context bases. A compares exact-zero with the registered mature multivariate
residual-AR lag sets. W and A each require guarded one-SE, at least 1%
improvement and three-of-four positive folds. Failure causes exact neutral
degradation and cannot invalidate a legal K route.

Only unscaled frozen-K Markov matrices enter the block Hankel. W and A residuals
are forbidden. ERA order is selected from the registered order list using
flight-grouped validation loss, one-SE, 2% regret and a strict stable-realization
gate. If no order passes, the honest output is
`MIMO_REALIZATION_NOT_SUPPORTED`; eigenvalues are not clipped after seeing data.

## Evidence boundary

This experiment can report predictive decomposition, stable Markov parameters
and a validation-supported realization. It cannot by itself establish true
rigid-body modes, unique aerodynamic laws, wind state, vortex-ring state or
cross-platform generalization. Those require independent excitation and
validation beyond this dataset.
