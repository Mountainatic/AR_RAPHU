# MULTISCALE-PHYSICS-AUDIT V1

Two-rod, bidirectional multirate physical-time audit. Raw data remain at their
native cadence. Each branch uses a predeclared cadence, target window, horizon,
history, and causal multiresolution lag basis.

## Scientific order

1. Preflight audits breakpoints and empirical time scales without fitting models.
2. Stage 1 scans linear baselines and linear channel kernels.
3. Stage 2 confirms at most two candidates per channel using 500 moving-block
   bootstrap replicates, common-support, placebo, kernel, regularization, and
   resolution diagnostics.
4. Stage 3 runs an exactly nested nonlinear amplitude expansion only for Stage-2
   passes.

No future input is used. PCA is fitted on the training rod only. Inner selection
uses four rolling-origin folds. Outer validation is bidirectional across rods.

## Run

```bash
export PYTHON=/root/AR_RAPHU_AUTODL/.venv/bin/python
bash RUN_SERVER.sh \
  --data /root/OPS_UOI_WORKSPACE/data/private/multiscale_physics_audit_v1/raw/data.xlsx \
  --sample-period-sec 2.0 \
  --n-jobs 12 \
  --bootstrap 500
```

The pipeline is idempotent at profile level. `RESUME_SERVER.sh` accepts the same
arguments and skips complete Stage-1 profiles whose data/config identities match.

## Privacy and packaging

`scripts/build_return_bundle.py` rejects Excel files, `.git`, caches, and temporary
files. It builds and independently re-verifies the manifest and every SHA256 after
extracting the ZIP to a fresh temporary directory.
