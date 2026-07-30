# PHYSICS-FIRST K → Residual-AR CPU Benchmark

This directory implements the frozen L6 CPU benchmark. It does not search the
time scale and it does not contain a Q layer.

## Frozen task

- cadence: 10 seconds
- history: 40 minutes
- horizon: 20 minutes
- target window: 2 minutes
- outer directions: Sheet1→Sheet2 and Sheet2→Sheet1
- inner selection: four rolling-origin folds with a 22-minute purge

The formal model is fitted in one direction only:

1. fit and freeze physical joint-lift K;
2. generate rolling OOF K residuals;
3. expose only residuals whose target window has matured before the current
   origin;
4. fit residual AR with an exact-zero candidate;
5. never jointly refit K and residual AR.

## uv environment

The repository `pyproject.toml` and `uv.lock` own all dependencies. On the
AutoDL image:

```bash
export VIRTUAL_ENV=/root/AR_RAPHU_AUTODL/.venv
UV=/root/AR_RAPHU_AUTODL/.autodl-tools/uv
$UV sync --active
```

All run scripts call `uv run --active`; direct `pip` execution is not part of
the reproducible protocol.

## Build the immutable shared package

```bash
$UV run --active python scripts/preflight.py \
  --data /root/OPS_UOI_WORKSPACE/data/private/multiscale_physics_audit_v1/raw/data.xlsx

$UV run --active python scripts/build_shared_dataset.py \
  --data /root/OPS_UOI_WORKSPACE/data/private/multiscale_physics_audit_v1/raw/data.xlsx
```

The generated shared ZIP is the only allowed CPU/GPU data handoff. GPU code
must not regenerate PCA, targets, masks, or sample IDs.

## Run or resume

```bash
bash RUN_CPU.sh --shared-data shared --n-jobs 20 --bootstrap 500
bash RESUME_CPU.sh --shared-data shared --n-jobs 20 --bootstrap 500
```

Input-only and dynamic models are written to separate leaderboards. Methods
without a reliable paper-faithful Python implementation are retained with an
explicit `ADAPTED_IMPLEMENTATION` label and a limitation note.
