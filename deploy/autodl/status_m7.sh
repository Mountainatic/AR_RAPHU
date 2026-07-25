#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_m7.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "M7 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "M7 process is not running."
fi

done_count="$(find results/phase1/job_records/AR-S1_G2_M7_candidates -type f -name DONE.json 2>/dev/null | wc -l)"
summary_count="$(find results/phase1/E2_AR-S1_G2/M7 -type f -name summary.json 2>/dev/null | wc -l)"
echo "M7 jobs: ${done_count}/90 DONE; candidate summaries=${summary_count}/90"
test -f results/phase1/E2_AR-S1_G2/M7/validation_selection.json \
  && echo "M7 validation selection: COMPLETED" \
  || echo "M7 validation selection: NOT_YET_RUN"
test -f results/phase1/E2_AR-S1_G2/M7/test_metrics.json \
  && echo "M7 test aggregation: COMPLETED" \
  || echo "M7 test aggregation: NOT_YET_RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_m7_log_path.txt; then
  log_file="$(cat results/runtime/autodl_m7_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 24 "${log_file}" 2>/dev/null || true
fi
