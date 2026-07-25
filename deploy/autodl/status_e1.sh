#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_e1.pid"

if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "E1 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "E1 process is not running."
fi

nvidia-smi

count_files() {
  local root="$1"
  local pattern="$2"
  find "${root}" -type f -name "${pattern}" 2>/dev/null | wc -l
}

echo "B0: $(count_files results/phase1/E1_AR-S0_G2/B0 test_metrics.json)/10"
echo "B1: $(count_files results/phase1/E1_AR-S0_G2/B1 test_metrics.json)/10"
echo "M4 dense AR: $(count_files results/phase1/E1_AR-S0_G2/Track-AR dense_summary.json)/10"
echo "M5 XAR warmup: $(count_files results/phase1/E1_AR-S0_G2/Track-XAR warmup.pt)/10"
echo "M5 XAR fork: $(count_files results/phase1/job_records/AR-S0_G2_XAR_fork DONE.json)/90"

if test -f results/phase1/E1_AR-S0_G2/Track-XAR/validation_selection.json; then
  echo "M5 validation selection: COMPLETED"
else
  echo "M5 validation selection: NOT_YET_RUN"
fi
if test -f results/phase1/E1_AR-S0_G2/Track-XAR/test_metrics.json; then
  echo "M5 test aggregation: COMPLETED"
else
  echo "M5 test aggregation: NOT_YET_RUN"
fi

if test -f results/runtime/autodl_e1_log_path.txt; then
  log_file="$(cat results/runtime/autodl_e1_log_path.txt)"
  if test -f "${log_file}"; then
    echo "Recent log: ${log_file}"
    tail -n 30 "${log_file}"
  fi
fi
