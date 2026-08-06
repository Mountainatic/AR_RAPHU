# PRISM v2.1.1 SRU implementation correction

This directory freezes the user-authorized v2.1.1 repair contract. It inherits
the v2.1 SRU theory and baseline replay amendment, preserves the complete C1
shared-data base, reuses the frozen v2.1 baseline predictions, and writes only
to `results_prism_v2_1_1_sru`.

The required order is:

```text
E0R -> E1R -> E2R-K -> E2R-C -> E3R-W -> E4R-A -> E5R-Joint
    -> E5.5 development decision -> E6R -> E7R -> E8R
```

E6R and later stages remain inaccessible unless the frozen development
continue gate passes. A failed gate is a valid terminal result and must retain
`test_accessed=false`.
