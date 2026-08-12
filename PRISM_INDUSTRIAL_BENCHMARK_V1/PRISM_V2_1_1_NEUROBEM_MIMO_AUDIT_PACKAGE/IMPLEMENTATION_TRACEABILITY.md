# NeuroBEM implementation traceability

| Contract | Code owner | Test/artifact |
|---|---|---|
| Parent-flight split isolation | `neurobem_data.py` | `test_parent_flight_is_atomic_across_splits`; `N1/SPLIT_MANIFEST.json` |
| Continuous-segment history isolation | `neurobem_data.py`, `neurobem_linear.py` | `test_k_design_is_strictly_lagged_and_segment_local`; `N1/SEGMENT_REGISTRY.json` |
| Test lockbox | `neurobem_data.py`, `neurobem_runner.py` | `test_locked_segment_cannot_be_read`; `N6_FREEZE/DEVELOPMENT_FREEZE.json` |
| Rigid-body generalized-force targets | `neurobem_data.py` | `test_generalized_torque_includes_rigid_body_cross_term`; `N2_K/RESULT.json` |
| Native-support causal MISO K | `neurobem_experiment.py` | `test_small_grouped_k_audit_uses_all_original_folds`; `N2_K/RESULT.json` |
| Numerical-only ridge certificate | `neurobem_linear.py` | `test_numerical_ridge_uses_smallest_certified_alpha_and_roundtrips`; model contracts |
| Frozen-K W correction | `neurobem_experiment.py` | identity equivalence in `N3_W/RESULT.json` |
| Mature residual A | `neurobem_experiment.py` | maturity/support fields in `N4_A/RESULT.json` |
| K-only Hankel/ERA | `neurobem_linear.py`, `neurobem_experiment.py` | `test_era_recovers_stable_markov_sequence`; `N5_MIMO_ERA/RESULT.json` |
| Route freeze before test | `neurobem_runner.py` | `N6_FREEZE/DEVELOPMENT_FREEZE.json` |
| One locked evaluation | `neurobem_runner.py` | `N7_TEST/RESULT.json`, `N8_FINAL/FINAL_RESULT.json` |

The official processed archive is not a repository artifact. Its byte size and
SHA256 are recorded at N0, and the distributed files are excluded from all Git
and return-package paths.
