# PF Independent Freeze Semantics Review

## Scope

Reviewed the v2.1.2 Metro runner, assembly, final materialization, reporting,
canonical v2.1.1 theory, and existing Joint OOF correction artifacts.

## Findings before patch

1. M5 treated different PF and Joint gate outcomes as an implementation
   inconsistency even when the evaluated predictions differed.
2. M6 required PF and Joint gate outcomes to match and therefore could not
   freeze an otherwise valid PF route.
3. M6 always registered PF and Joint pending candidate IDs.
4. M7 always read Joint development results, refit all Joint contracts, and
   produced Joint test/OOD predictions.
5. M8 assumed Joint predictions and comparisons always existed.

## Implemented contract

- PF mandatory checks and Joint optional checks are evaluated separately.
- Legal protocol/numerical/binding evidence plus a Joint model-gate failure
  produces `PASS_PF_ONLY`, not a global stop.
- Protocol, route-materialization, true-joint-fit, numerical, and binding
  failures remain hard stops.
- Candidate IDs, M7 estimator reads/materialization, and M8 comparisons are
  controlled by the M6 `formal_routes` manifest.
- Same gate contract applied to different predictions may produce different
  outcomes. Only the same complete evaluation identity with contradictory
  outcomes is an inconsistency.

## Non-changes

No estimator file, candidate set, fold, cap, ridge grid, one-SE rule,
activation threshold, input-path threshold, numerical certificate, or data
partition was changed. M2--M5 development values are reused unchanged. No
test/OOD access is part of this amendment.
