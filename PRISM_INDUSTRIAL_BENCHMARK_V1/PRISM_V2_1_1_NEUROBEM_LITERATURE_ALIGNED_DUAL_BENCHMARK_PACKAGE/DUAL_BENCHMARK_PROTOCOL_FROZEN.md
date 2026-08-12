# PRISM v2.1.1 NeuroBEM Literature-Aligned Dual Benchmark R1

Status: `FROZEN_BEFORE_MODEL_IMPLEMENTATION_AND_NEW_TEST_ACCESS`.

This prospective extension begins at reporting commit
`48fea5e03eab89a327dfcf6b07f56e79614c575d`. It does not overwrite either
completed NeuroBEM result namespace. Published baseline values were already
public before this extension, are reporting references only, and are forbidden
from estimator or hyperparameter selection.

## Shared sequence

Literature audit and theory-interface repair precede implementation. Both
tracks then complete development. A single global freeze binds both tracks
before either PRISM test partition is read. After that access, only reporting
code and artifacts may change. A target, split, leakage, or metric defect found
after access invalidates this experiment ID; it is not silently repaired.

## Track A

Track A uses the NeuroBEM official 400 Hz processed data and official
`testset.txt`. The target is the six-dimensional body-frame total force/torque,
formed as official base-model output plus residual target. History is fixed at
20 samples (50 ms). The comparable input set is body linear velocity, body
angular velocity, and four motor speeds. `A` is disabled.

The formal routes are `PRISM_PF_KC`, `PRISM_PF_KCW`, `PRISM_J_KC`, and
`PRISM_J_KCW`. `K` is actuator history. Motion variables in `K` are never given
a causal actuator label. Cross-context terms belong to `C`; `W` receives only
the frozen `C` latent. The published metric gate must be reproduced from the
official NeuroBEM prediction files before exact direct-comparison language is
allowed.

## Track B

Track B uses the official long-horizon release and its `train`, `valid`, and
`test` directories. The official preprocessor resets each trajectory time and
applies a 0.01-second mean resample. `H=20`, `U=10`, and `T=60` are fixed.
Motor speeds are multiplied by `1e-3`, as in the official implementation.

The learned state excludes position and contains linear velocity, unit
quaternion, and body angular velocity. Future registered motor commands are
available during rollout. Future measured states and target residuals are not.
Velocity prediction uses the six-dimensional increment of
`z=[v, omega]`; attitude prediction uses a rotation-vector increment, maps it
through the quaternion exponential, composes it with the current unit
quaternion, and renormalizes.

The official decoupled reference implementation teacher-forces the
complementary state branch during recursion. This experiment intentionally
obeys the stricter registered no-future-state contract. Therefore an otherwise
matching result must retain the information-set compatibility qualification.
`A` is disabled. Formal routes are `PRISM_KC`, `PRISM_KCW`, `PRISM_J_KC`, and
`PRISM_J_KCW`.

## W theory interface

Canonical `W` candidates are identity, natural-cubic latent curvature, and
signed-quadratic latent curvature. The prior speed/aerodynamic-context `W`
implementation remains immutable historical evidence and is classified only
as `AERODYNAMIC_CONTEXT_W_EXTENSION_DIAGNOSTIC`.
