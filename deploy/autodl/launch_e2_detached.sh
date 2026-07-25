#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs results/runtime

PID_FILE="results/runtime/autodl_e2.pid"
if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "E2 is already running with PID $(cat "${PID_FILE}")." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/autodl_e2_${STAMP}.log"
nohup setsid bash deploy/autodl/run_e2_resume.sh >"${LOG_FILE}" 2>&1 </dev/null &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "${LOG_FILE}" >results/runtime/autodl_e2_log_path.txt
echo "Started E2 in background: PID=${PID} log=${LOG_FILE}"
