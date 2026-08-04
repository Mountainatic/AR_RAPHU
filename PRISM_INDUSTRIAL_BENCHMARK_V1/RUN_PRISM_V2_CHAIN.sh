#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="${SHARED:-/root/autodl-tmp/PRISM_SHARED_DATA_C1}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/PRISM_V2_MODULAR_CPU_RESULTS}"
PYTHON="${PYTHON:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"
mkdir -p "$OUTPUT/logs"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_stage() {
  local stage="$1" jobs="$2"
  local log="$OUTPUT/logs/${stage^^}.log"
  printf '%s START %s jobs=%s\n' "$(date --iso-8601=seconds)" "$stage" "$jobs" | tee -a "$OUTPUT/logs/CHAIN.log"
  "$PYTHON" "$PROJECT/scripts/run_prism_v2_stage.py" "$stage" \
    --shared "$SHARED" --project "$PROJECT" --output "$OUTPUT" --n-jobs "$jobs" \
    2>&1 | tee "$log"
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$stage" | tee -a "$OUTPUT/logs/CHAIN.log"
}

run_stage v1 20
run_stage v2 28
run_stage v3 6
run_stage v4 5
run_stage v5 10
run_stage v6 1
run_stage v7 5
run_stage g3 1
printf '%s DEVELOPMENT_CHAIN_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"

