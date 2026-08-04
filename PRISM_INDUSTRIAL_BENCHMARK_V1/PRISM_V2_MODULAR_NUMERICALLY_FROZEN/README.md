# PRISM v2 Modular Theory Bundle — Numerically Frozen

This bundle contains the PRISM v2 modular theory and the numerically frozen CPU benchmark protocol.

## Files

- `PRISM_Theory_v2_0_Modular_Assembly_Theory_Only.md`  
  Modular operator theory: two faces, E/K/C/W/A/J interfaces, neutral elements, ownership boundaries and certification.

- `PRISM_V2_MODULAR_CPU_BENCHMARK_PROTOCOL.md`  
  CPU-only benchmark using the existing five datasets, seven tasks and immutable C0/C1 splits.

- `PRISM_V2_ASSEMBLY_CONFIG_FROZEN.json`  
  Machine-readable single source of truth for every implementation-facing numeric threshold.

- `PRISM_V2_NUMERICAL_FREEZE.md`  
  Human-readable formulas, grids, activation gates, stability rules, numerical certificates, statistical support, OOD and stopping semantics.

- `PRISM_V2_NUMERICAL_FREEZE_AUDIT.json`  
  Audit proving that the implementation-facing freeze has no unresolved numeric field.

- `PRISM_V2_CHANGELOG_FROM_V1_3.md`  
  Conceptual and numerical migration from v1.3 to v2.0.

- `PRISM_V2_FROZEN_DATA_INHERITANCE.json`  
  Machine-readable task and split inheritance record.

- `MANIFEST.json` and `SHA256SUMS.txt`  
  Bundle integrity records.

## Status

- Theory: `DRAFT_FOR_REVIEW`
- Numerical config: `FROZEN_BEFORE_IMPLEMENTATION_AND_V2_DEVELOPMENT_ACCESS`
- Primary-head v2 evaluation: `POST_HOC_EXPLORATORY`
- Unaccessed registered views: `PROSPECTIVE_INTERNAL_CONFIRMATION`
- GPU benchmark: `DEFERRED`

Implementation rule: when a required numeric value is absent, stop and report the missing field. Never add a default.
