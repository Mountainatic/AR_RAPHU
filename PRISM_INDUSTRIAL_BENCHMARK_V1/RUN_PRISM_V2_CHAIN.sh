#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="${SHARED:-/root/autodl-tmp/PRISM_SHARED_DATA_C1}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/PRISM_V2_MODULAR_CPU_RESULTS}"
PYTHON="${PYTHON:-/root/AR_RAPHU_AUTODL/.venv/bin/python}"
mkdir -p "$OUTPUT/logs"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export PRISM_MEMORY_RESERVE_GIB="${PRISM_MEMORY_RESERVE_GIB:-4}"

run_stage() {
  local stage="$1" jobs="$2"
  local log="$OUTPUT/logs/${stage^^}.log"
  printf '%s START %s jobs=%s\n' "$(date --iso-8601=seconds)" "$stage" "$jobs" | tee -a "$OUTPUT/logs/CHAIN.log"
  "$PYTHON" "$PROJECT/scripts/run_prism_v2_stage.py" "$stage" \
    --shared "$SHARED" --project "$PROJECT" --output "$OUTPUT" --n-jobs "$jobs" \
    2>&1 | tee "$log"
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$stage" | tee -a "$OUTPUT/logs/CHAIN.log"
}

run_stage v1 31
run_stage v2 31
run_stage v3 31
run_stage v4 31
run_stage v5 31
run_stage v6 1
run_stage v7 31
run_stage bdev 31
run_stage g3 1
run_stage v8c 31
run_stage v8b 31
printf '%s START report\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"
"$PYTHON" "$PROJECT/scripts/run_prism_v2_stage.py" report \
  --shared "$SHARED" --project "$PROJECT" --output "$OUTPUT" --n-jobs 1 \
  --c6-summary /root/autodl-tmp/PRISM_V2_BASELINE_CACHE/C6_V2_SUMMARY.tar \
  2>&1 | tee "$OUTPUT/logs/REPORT.log"
printf '%s COMPLETE report\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"
printf '%s START package\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"
mkdir -p /root/autodl-tmp/PRISM_V2_MODULAR_CPU_RELEASE
"$PYTHON" "$PROJECT/scripts/build_prism_v2_release.py" \
  --project "$PROJECT" --output "$OUTPUT" \
  --return-dir /root/autodl-tmp/PRISM_V2_MODULAR_CPU_RELEASE \
  2>&1 | tee /root/autodl-tmp/PRISM_V2_MODULAR_CPU_RELEASE/PACKAGE.log
printf '%s COMPLETE package\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"
printf '%s DEVELOPMENT_CHAIN_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/CHAIN.log"
