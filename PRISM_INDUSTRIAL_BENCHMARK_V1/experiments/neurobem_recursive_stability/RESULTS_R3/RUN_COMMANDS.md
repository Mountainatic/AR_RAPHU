# Exact R3 commands

All commands ran from `/root/autodl-tmp/PRISM_V211_NEUROBEM_MANIFOLD_SWITCH_R1/PRISM_INDUSTRIAL_BENCHMARK_V1` with `/root/AR_RAPHU_AUTODL/.venv/bin/python`, `PYTHONPATH=src`, 24 ordered Linux-fork workers, and OMP/MKL/OpenBLAS/NumExpr threads fixed to 1.

Calibration:

```bash
python -m experiments.neurobem_recursive_stability.run_experiment --stage calibration --config experiments/neurobem_recursive_stability/configs/calibration.yaml --r2-model /root/autodl-tmp/PRISM_V211_NEUROBEM_MANIFOLD_SWITCH_RUNTIME/r2_formal_test/PRISM_V2_1_1_NEUROBEM_MANIFOLD_SWITCH_R2_20260813T105419/checkpoints/global_prism_r2_train_only.json --data-root /root/autodl-tmp/PRISM_V211_NEUROBEM_MIMO/data/private/neurobem --release-root /root/autodl-tmp/PRISM_V211_NEUROBEM_LITERATURE_RUNTIME/track_b_release --split-manifest /root/autodl-tmp/PRISM_V211_NEUROBEM_MIMO/PRISM_INDUSTRIAL_BENCHMARK_V1/results_prism_v2_1_1_neurobem_exact_published_training/TRACK_B_SPLIT_MANIFEST.json --output-root /root/autodl-tmp/PRISM_V211_NEUROBEM_RECURSIVE_STABILITY_R3_RUNTIME/calibration_final
```

Formal held-out test (run once from generating commit `3562d71d183603b5c12332a8bdcb689762f59060`):

```bash
python -m experiments.neurobem_recursive_stability.run_experiment --stage test --config experiments/neurobem_recursive_stability/configs/calibration.yaml --calibration-freeze experiments/neurobem_recursive_stability/CALIBRATION_FREEZE/R3_CALIBRATION_FREEZE.json --r2-model /root/autodl-tmp/PRISM_V211_NEUROBEM_MANIFOLD_SWITCH_RUNTIME/r2_formal_test/PRISM_V2_1_1_NEUROBEM_MANIFOLD_SWITCH_R2_20260813T105419/checkpoints/global_prism_r2_train_only.json --data-root /root/autodl-tmp/PRISM_V211_NEUROBEM_MIMO/data/private/neurobem --release-root /root/autodl-tmp/PRISM_V211_NEUROBEM_LITERATURE_RUNTIME/track_b_release --split-manifest /root/autodl-tmp/PRISM_V211_NEUROBEM_MIMO/PRISM_INDUSTRIAL_BENCHMARK_V1/results_prism_v2_1_1_neurobem_exact_published_training/TRACK_B_SPLIT_MANIFEST.json --output-root /root/autodl-tmp/PRISM_V211_NEUROBEM_RECURSIVE_STABILITY_R3_RUNTIME/formal_test
```
