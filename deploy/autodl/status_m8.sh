#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_m8.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "M8 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "M8 process is not running."
fi
done_count="$(find results/phase1/job_records/AR-S1_G2_M8_candidates -type f -name DONE.json 2>/dev/null | wc -l)"
summary_count="$(find results/phase1/E2_AR-S1_G2/M8 -type f -name summary.json 2>/dev/null | wc -l)"
echo "M8 jobs: ${done_count}/90 DONE; candidate summaries=${summary_count}/90"
test -f results/phase1/E2_AR-S1_G2/M8/validation_selection.json \
  && echo "M8 validation selection: COMPLETED" \
  || echo "M8 validation selection: NOT_YET_RUN"
test -f results/phase1/E2_AR-S1_G2/M8/rank_audit.json \
  && echo "M8 rank audit: COMPLETED" \
  || echo "M8 rank audit: NOT_YET_RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_m8_log_path.txt; then
  log_file="$(cat results/runtime/autodl_m8_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 24 "${log_file}" 2>/dev/null || true
fi
