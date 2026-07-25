#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_m6.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "M6 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "M6 process is not running."
fi

if test -d results/phase1/job_records/AR-S2_G2_M6_candidates; then
  done_count="$(find results/phase1/job_records/AR-S2_G2_M6_candidates -type f -name DONE.json | wc -l)"
else
  done_count=0
fi
if test -d results/phase1/E3_AR-S2_G2/M6; then
  summary_count="$(find results/phase1/E3_AR-S2_G2/M6 -type f -name summary.json | wc -l)"
else
  summary_count=0
fi
echo "M6 jobs: ${done_count}/40 DONE; candidate summaries=${summary_count}/40"
test -f results/phase1/E3_AR-S2_G2/M6/validation_selection.json \
  && echo "M6 validation selection: COMPLETED" \
  || echo "M6 validation selection: NOT_YET_RUN"
test -f results/phase1/E3_AR-S2_G2/M6/test_metrics.json \
  && echo "M6 test aggregation: COMPLETED" \
  || echo "M6 test aggregation: NOT_YET_RUN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if test -f results/runtime/autodl_m6_log_path.txt; then
  log_file="$(cat results/runtime/autodl_m6_log_path.txt)"
  echo "Recent log: ${log_file}"
  tail -n 24 "${log_file}" 2>/dev/null || true
fi
