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

run_stage_once() {
  local stage="$1" jobs="$2"
  local log="$OUTPUT/logs/${stage^^}.log"
  printf '%s START %s jobs=%s\n' "$(date --iso-8601=seconds)" "$stage" "$jobs" | tee -a "$OUTPUT/logs/CHAIN.log"
  if "$PYTHON" "$PROJECT/scripts/run_prism_v2_stage.py" "$stage" \
      --shared "$SHARED" --project "$PROJECT" --output "$OUTPUT" --n-jobs "$jobs" \
      2>&1 | tee "$log"; then
    printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$stage" | tee -a "$OUTPUT/logs/CHAIN.log"
    return 0
  else
    local status=$?
    printf '%s FAILED %s exit=%s\n' "$(date --iso-8601=seconds)" "$stage" "$status" | tee -a "$OUTPUT/logs/CHAIN.log"
    return "$status"
  fi
}

run_stage_retry() {
  local stage="$1" jobs="$2" attempts="$3"
  local attempt
  for ((attempt=1; attempt<=attempts; attempt++)); do
    printf '%s ATTEMPT %s %s/%s\n' "$(date --iso-8601=seconds)" "$stage" "$attempt" "$attempts" | tee -a "$OUTPUT/logs/CHAIN.log"
    if run_stage_once "$stage" "$jobs"; then
      return 0
    fi
    if (( attempt < attempts )); then
      printf '%s RETRY %s checkpoint-preserving\n' "$(date --iso-8601=seconds)" "$stage" | tee -a "$OUTPUT/logs/CHAIN.log"
      sleep 5
    fi
  done
  return 1
}

run_stage_retry v1 31 2
run_stage_retry v2 19 3
run_stage_retry v3 31 2
run_stage_retry v4 31 2
run_stage_retry v5 31 2
run_stage_retry v6 1 2
run_stage_retry v7 31 2
run_stage_retry bdev 31 2
run_stage_retry g3 1 2
run_stage_retry v8c 31 2
run_stage_retry v8b 31 2
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
