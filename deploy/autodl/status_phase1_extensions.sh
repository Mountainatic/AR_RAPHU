#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_phase1_extensions.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Phase-1 extensions running: PID=$(cat "${PID_FILE}")"
else
  echo "Phase-1 extensions not running."
fi

count_done() {
  local directory="$1"
  if test -d "${directory}"; then
    find "${directory}" -type f -name DONE.json | wc -l
  else
    echo 0
  fi
}

for scenario in AR-S4 AR-S5 AR-S6 AR-S7; do
  warmup="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_warmup")"
  forks="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_fork")"
  root="results/phase1/SUPPORT_${scenario}_${scenario}_G2/Track-XAR"
  selection="NOT_YET_RUN"
  test_status="NOT_YET_RUN"
  test -f "${root}/validation_selection.json" && selection="COMPLETED"
  test -f "${root}/test_metrics.json" && test_status="COMPLETED"
  echo "${scenario}: warmup=${warmup}/10 forks=${forks}/90 selection=${selection} test=${test_status}"
done
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_phase1_extensions_log_path.txt; then
  log_file="$(cat results/runtime/autodl_phase1_extensions_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 20 "${log_file}" 2>/dev/null || true
fi
