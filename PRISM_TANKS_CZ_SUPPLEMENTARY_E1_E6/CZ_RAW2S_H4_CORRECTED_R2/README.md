# Private CZ raw-2s H4 corrected E1-E6 evidence

Status: `COMPLETED`.

This directory contains public, aggregate-only evidence from the corrected
private CZ rerun. The task is sampled every 2 seconds with L256 and
H4/W1/W0=1. Inputs are `[t-256,t)`, the anchor is `D[t-1]`, and the target
is `D[t+3]-D[t-1]`, so the formal horizon is 8 seconds.

The run used execution commit `6de39fedbba18d3fbc4333d5cd250d10fbd263e3`,
which is a descendant of the strict-OOF authority commit
`2ee6273b8f915cbcdff2f46d56bc80047ddae4a7`. The dependency purge was
`256 + 4 = 260` samples. Selection was frozen before formal-target metrics
were accessed.

No raw workbook, sample-level rows, prediction arrays, fitted checkpoints,
or private filesystem paths are included here. The workbook is represented
only by SHA-256 in `SANITIZED_PROVENANCE.json`.

E1's repeated IDENTITY/K/KC/KCW development metrics are expected, not copied
results: strict nested OOF rejected K, C, and W as zero-identity additions in
both directions. Only A was admitted in the dynamic information set. The
complete stage decisions and fold-level losses are retained in the two
`*_dynamic_routes.json` files.
