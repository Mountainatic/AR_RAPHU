#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs results/runtime
PID_FILE="results/runtime/autodl_m8.pid"
LOG_PATH_FILE="results/runtime/autodl_m8_log_path.txt"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "M8 is already running: PID=$(cat "${PID_FILE}")"
  exit 0
fi
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="logs/autodl_m8_${timestamp}.log"
nohup bash deploy/autodl/run_m8_resume.sh >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${PID_FILE}"
echo "${log_file}" >"${LOG_PATH_FILE}"
echo "Started M8 in background: PID=${pid} log=${log_file}"
