# CZ raw-2s H4 authority E1 execution report (2026-09-23)

## Outcome

E1 completed with `PASS` for both independent directions on the registered
`CZ_DIAM_RAW2S_CURRENT_L256_H4` task:

- sampling interval: 2 seconds;
- input: `[t-256,t)`;
- anchor: `D[t-1]`;
- target: `D[t+3]-D[t-1]`;
- `H/W/W0 = 4/1/1`, i.e. an 8-second forecast;
- development selection remained frozen and formal-test metrics had no
  selection authority.

The sealed development anchor was
`/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_20260922_R3`.  The corrected
E1 output is
`/root/autodl-tmp/PRISM_CZ_RAW2S_H4_AUTHORITY_E1_E6_EXEC_20260923_R3`.
The output is about 20 MiB and does not contain the private raw workbook.

## Authority correction discovered during execution

The authority branch used the registered family name
`BEST_ACTIVE_K_CHANNEL` in C selection, but the portable final-checkpoint
fitter and predictor compared it with the non-existent literal
`BEST_ACTIVE_K`.  Consequently the old R3 final checkpoint incorrectly routed
the zero-C identity parent through a one-feature Ridge refit.

The correction imports and uses the existing `v211_c.BEST_ACTIVE_K` constant in
both final fitting and inference.  It changes neither candidates nor OOF
selection.  The authority audit pins the original blob and the reviewed patched
blob separately.

## Formal stagewise results

| Direction | Stage | Delta RMSE | Delta R2 | Reconstructed-level R2 |
|---|---:|---:|---:|---:|
| Rod1 to Rod2 | K | 0.0152383309 | 0.0935035568 | 0.9982935852 |
| Rod1 to Rod2 | K+C | 0.0152383309 | 0.0935035568 | 0.9982935852 |
| Rod1 to Rod2 | K+C+Delta-W | 0.0152383309 | 0.0935035568 | 0.9982935852 |
| Rod1 to Rod2 | K+C+Delta-W+A | 0.0138185101 | 0.2545581903 | 0.9985967591 |
| Rod1 to Rod2 | Joint | 0.0137119508 | 0.2660105884 | 0.9986183174 |
| Rod2 to Rod1 | K | 0.0166050730 | 0.0779871352 | 0.9991607446 |
| Rod2 to Rod1 | K+C | 0.0166050730 | 0.0779871352 | 0.9991607446 |
| Rod2 to Rod1 | K+C+Delta-W | 0.0166050730 | 0.0779871352 | 0.9991607446 |
| Rod2 to Rod1 | K+C+Delta-W+A | 0.0152233212 | 0.2250490352 | 0.9992946066 |
| Rod2 to Rod1 | Joint | 0.0148219284 | 0.2653764865 | 0.9993313144 |

## Acceptance evidence

- 31 relevant tests passed on the server after the correction.
- Pure-K portable-checkpoint reload maximum error: `0.0` in both directions.
- C nested-OOF identity fold-loss maximum error: `0.0` in both directions.
- Formal pure-K versus K+C maximum prediction error: `0.0` in both directions.
- Formal K+C versus K+C+Delta-W maximum prediction error: `0.0` in both
  directions.
- Every prefix in one direction used the same scoring-support hash and sample
  order.
- The sealed R3 anchor was not modified.

## E2--E6 gate

E2--E6 were not started.  The mandatory private-storage gate was executed and
returned `BLOCKED`: 13.295 GiB was free versus the frozen 300 GiB requirement.
Private CZ artifacts were not redirected to the public mount.  The E2--E6
campaign must resume only after at least 300 GiB of private writable storage is
attached.

## Code and execution bindings

- public branch: `prism-unified-hw-r2-protocols-20260921`;
- branch implementation commit used for the final E1 logic: `31dd54d`;
- server cherry-pick equivalent: `e623fe993f85da7843df724167305d51f44c7476`;
- authority ancestor: `2ee6273b8f915cbcdff2f46d56bc80047ddae4a7`.
