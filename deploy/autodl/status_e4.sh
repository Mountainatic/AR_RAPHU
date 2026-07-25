#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_e4.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "E4 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "E4 process is not running."
fi

count_files() {
  local directory="$1"
  local pattern="$2"
  if test -d "${directory}"; then
    find "${directory}" -type f -name "${pattern}" | wc -l
  else
    echo 0
  fi
}

warmup_done="$(count_files results/phase1/job_records/AR-S3_G2_XAR_warmup DONE.json)"
fork_done="$(count_files results/phase1/job_records/AR-S3_G2_XAR_fork DONE.json)"
echo "E4 M5 jobs: warmup=${warmup_done}/10; forks=${fork_done}/90"
test -f results/phase1/E4_AR-S3_G2/Track-XAR/validation_selection.json \
  && echo "E4 validation selection: COMPLETED" \
  || echo "E4 validation selection: NOT_YET_RUN"
test -f results/phase1/E4_AR-S3_G2/Track-XAR/test_metrics.json \
  && echo "E4 test aggregation: COMPLETED" \
  || echo "E4 test aggregation: NOT_YET_RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_e4_log_path.txt; then
  log_file="$(cat results/runtime/autodl_e4_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 24 "${log_file}" 2>/dev/null || true
fi
