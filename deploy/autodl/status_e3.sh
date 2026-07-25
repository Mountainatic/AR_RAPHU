#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_e3.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "E3 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "E3 process is not running."
fi
count_done() {
  local root="$1"
  if test -d "${root}"; then
    find "${root}" -type f -name DONE.json | wc -l
  else
    echo 0
  fi
}
warmup="$(count_done results/phase1/job_records/AR-S2_G2_XAR_warmup)"
forks="$(count_done results/phase1/job_records/AR-S2_G2_XAR_fork)"
echo "E3 M5 XAR warmup: ${warmup}/10; forks: ${forks}/90"
test -f results/phase1/E3_AR-S2_G2/Track-XAR/validation_selection.json \
  && echo "E3 M5 validation selection: COMPLETED" \
  || echo "E3 M5 validation selection: NOT_YET_RUN"
test -f results/phase1/E3_AR-S2_G2/Track-XAR/test_metrics.json \
  && echo "E3 M5 test aggregation: COMPLETED" \
  || echo "E3 M5 test aggregation: NOT_YET_RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_e3_log_path.txt; then
  log_file="$(cat results/runtime/autodl_e3_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 24 "${log_file}" 2>/dev/null || true
fi
