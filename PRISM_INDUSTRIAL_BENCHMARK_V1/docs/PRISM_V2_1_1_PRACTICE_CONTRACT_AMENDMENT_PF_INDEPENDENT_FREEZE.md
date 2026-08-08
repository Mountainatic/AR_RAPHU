# PRISM v2.1.1 Practice Contract Amendment: PF Independent Freeze

Status: practice/execution contract clarification for PRISM v2.1.2.

This amendment does not define a new estimator version. It does not change K,
C, W, A, or Joint estimation; candidate families; hyperparameters; one-SE;
row caps; inner folds; data partitions; numerical certificates; or any gate
threshold.

## Route semantics

Physical-First is the mandatory structured route:

```text
K -> C -> W -> A
```

Joint is an optional predictive enhancement route over the registered
`J_K`, `J_KW`, `J_KA`, and `J_KWA` family. Its development evidence is
independent of the validity of an already legal Physical-First assembly.

Consequently:

```text
PF_VALID does not imply JOINT_VALID
JOINT_INVALID does not imply PF_INVALID
```

When PF passes all mandatory development contracts and Joint passes its own
protocol, numerical, binding, and predictive gates, M6 freezes both routes.
When PF passes and Joint's protocol/numerics/binding are legal but its own
development predictive gate fails, M6 freezes PF only and records Joint as
`JOINT_NOT_SUPPORTED_ON_DEVELOPMENT` with diagnostic-only artifacts.

Protocol mismatch, missing Joint routes, non-joint W coefficients, numerical
failure, or candidate-binding failure remain hard stops. A PF contract failure
also remains a hard stop before test/OOD.

## Test/OOD eligibility

The M6 freeze manifest is authoritative:

- `formal_routes=["PHYSICS_FIRST"]` registers only `KC`, `KCW`, `KCA`,
  `KCWA`, and `PF_SELECTED` for later materialization.
- `formal_routes=["PHYSICS_FIRST","JOINT"]` additionally registers the four
  Joint routes and `J_SELECTED`.

M7 and M8 must never infer eligibility from the configured superset. They use
only `formal_routes`. A development-unsupported Joint route cannot be fitted,
predicted, scored, or restored after test/OOD access.

## Shared input gate

PF and Joint share one implementation, one configuration, and one set of
numerical semantics. They may pass or fail differently when the evaluated
predictions differ. `PF_JOINT_INPUT_GATE_INCONSISTENT` is reserved for the
case where gate version, parameters, prediction hash, target hash, and best-K
comparator hash are identical but the returned outcome differs.

This amendment is derived without new test/OOD access and cannot be used to
alter the already generated development predictions or losses.
