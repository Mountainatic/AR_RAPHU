#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_e2.pid"

if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "E2 process is running: PID=$(cat "${PID_FILE}")"
else
  echo "E2 process is not running."
fi
nvidia-smi
for stem in AR-S1_G2_X_fork AR-S1_G2_XAR_fork; do
  count="$(find "results/phase1/job_records/${stem}" -name DONE.json -type f 2>/dev/null | wc -l)"
  echo "${stem}: ${count}/90 DONE records"
done
if test -f results/runtime/autodl_e2_log_path.txt; then
  log_file="$(cat results/runtime/autodl_e2_log_path.txt)"
  if test -f "${log_file}"; then
    tail -n 40 "${log_file}"
  fi
fi
