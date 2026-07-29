#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON=/root/AR_RAPHU_AUTODL/.venv/bin/python
ROOT=results/cz_real_data/complete_5090
LOGS="$ROOT/logs"
mkdir -p "$LOGS" environment

"$PYTHON" -m pytest -q \
  > "$LOGS/00_full_regression_after_orss.log" 2>&1

"$PYTHON" tools/run_cz_r3.py \
  --stage history \
  --config configs/cz_real_data/r3_history.yaml \
  --solver orss --device cuda --furnace-a-only \
  --checkpoint-every-task --resume \
  2>&1 | tee "$LOGS/03_history.log"

"$PYTHON" tools/run_cz_r3.py \
  --stage resolution-penalty \
  --config configs/cz_real_data/r3_resolution.yaml \
  --solver orss --device cuda --furnace-a-only \
  --checkpoint-every-task --resume \
  2>&1 | tee "$LOGS/04_resolution_penalty.log"

"$PYTHON" tools/run_cz_r3.py \
  --stage continuation \
  --config configs/cz_real_data/r3_continuation.yaml \
  --solver orss --device cuda --furnace-a-only \
  --checkpoint-every-task --resume \
  2>&1 | tee "$LOGS/05_continuation.log"

"$PYTHON" tools/run_cz_r3.py \
  --stage rank \
  --config configs/cz_real_data/r3_rank.yaml \
  --solver orss --device cuda --furnace-a-only \
  --checkpoint-every-task --resume \
  2>&1 | tee "$LOGS/06_rank.log"

"$PYTHON" tools/run_cz_r4_baselines.py \
  --config configs/cz_real_data/r4_baselines.yaml \
  --furnace-a-only --resume \
  2>&1 | tee "$LOGS/07_baselines.log"

"$PYTHON" tools/freeze_cz_model.py \
  --results-root "$ROOT" \
  --output "$ROOT/frozen_model" \
  2>&1 | tee "$LOGS/08_freeze_model.log"

"$PYTHON" tools/run_cz_r5_confirmation.py \
  --config configs/cz_real_data/r5_confirmation.yaml \
  --locked-model "$ROOT/frozen_model" \
  --resume \
  2>&1 | tee "$LOGS/09_furnace_a_confirmation.log"

"$PYTHON" tools/run_cz_r6_outer.py \
  --config configs/cz_real_data/r6_outer.yaml \
  --frozen-model "$ROOT/frozen_model" \
  --zero-shot --resume \
  2>&1 | tee "$LOGS/10_furnace_b_zero_shot.log"

"$PYTHON" tools/run_cz_r7_calibration.py \
  --config configs/cz_real_data/r7_calibration.yaml \
  --frozen-model "$ROOT/frozen_model" \
  --fractions 0.05 0.10 --resume \
  2>&1 | tee "$LOGS/11_furnace_b_calibration.log"

"$PYTHON" tools/run_cz_interpretability.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --results-root "$ROOT" --resume \
  2>&1 | tee "$LOGS/12_interpretability.log"

"$PYTHON" tools/run_cz_bootstrap.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --results-root "$ROOT" \
  --development-replicates 250 \
  --confirmation-replicates 1000 \
  --resume \
  2>&1 | tee "$LOGS/13_bootstrap.log"

"$PYTHON" tools/summarize_cz_complete.py \
  --results-root "$ROOT" \
  --output CZ_COMPLETE_5090_REPORT.md \
  2>&1 | tee "$LOGS/14_summary.log"

{
  "$PYTHON" -V
  "$PYTHON" -c \
    'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0))'
  nvidia-smi
  git rev-parse HEAD
} > environment/CZ_5090_RUNTIME.txt

git add src tools tests configs environment \
  CZ_COMPLETE_5090_REPORT.md \
  CZ_COMPLETE_5090_STATUS.json
git add -f "$ROOT"
git commit -m "Add complete CZ 5090 ORSS experiment results"

bash tools/package_cz_complete.sh \
  2>&1 | tee "$LOGS/15_package.log"

echo "CZ_5090_COMPLETE_PIPELINE_FINISHED"
