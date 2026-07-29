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
S7_EXIT="$LOGS/01_orss_equivalence.exit"

mkdir -p "$LOGS"

while [[ ! -f "$S7_EXIT" ]]; do
  sleep 15
done

if [[ "$(tr -d '[:space:]' < "$S7_EXIT")" != "0" ]]; then
  echo "S7 equivalence failed; downstream launch is blocked."
  exit 70
fi

"$PYTHON" tools/run_cz_r2_1_audit.py \
  --config configs/cz_real_data/orss_5090.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --strict \
  --resume \
  > "$LOGS/02_r2_1.log" 2>&1
echo "$?" > "$LOGS/02_r2_1.exit"

"$PYTHON" tools/run_cz_r3.py \
  --stage history \
  --config configs/cz_real_data/r3_history.yaml \
  --solver orss \
  --device cuda \
  --furnace-a-only \
  --checkpoint-every-task \
  --resume \
  > "$LOGS/03_history.log" 2>&1
echo "$?" > "$LOGS/03_history.exit"
