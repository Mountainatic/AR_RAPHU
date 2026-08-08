# Changelog

## PRISM v2.1.2 - Joint OOF Protocol Correction

- Preserves the PRISM v2.1.1 Metro-P60 failed-stop branch, release, and result
  namespace as immutable formal audit evidence.
- Does not modify the W degradation/identity mechanism. Identity W remains
  exactly equivalent to skipping W.
- Does not modify K, C, or A model semantics, families, supports, profiles, or
  selection rules.
- Does not modify the Joint candidate family, ridge/penalty grids, one-SE rule,
  practical activation gate, positive-fold requirement, input-path thresholds,
  coefficient threshold, or numerical certificate.
- Corrects only the Joint development fold support. v2.1.1 used prior OOF
  validation chunks as the next Joint fit pool; v2.1.2 uses each original
  registered inner train/evaluation pair.
- Restricts W `PHYSICAL_OOF.parquet` to cross-stage fold/provenance auditing;
  it is never a Joint training pool.
- Adds strict C/W/Joint fold-ID hashes, complete four-fold coverage, fold-local
  C latent/W basis construction, and train-only inner A target-state access.
- Adds a diagnostic failure class so an MSE-only preservation failure is
  reported as `INPUT_PATH_PRESERVATION_PERFORMANCE_GATE_FAILED`, without
  changing PASS/FAIL semantics.
- Test and OOD data were not accessed before this correction and are not used
  to motivate or tune it.
