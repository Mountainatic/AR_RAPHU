#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
PID_FILE="results/runtime/autodl_e2.pid"

if test -s "${PID_FILE}" && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  PID="$(cat "${PID_FILE}")"
  kill -TERM -- "-${PID}"
  echo "Sent TERM to E2 process group ${PID}."
else
  echo "No active E2 process found."
fi
bash deploy/autodl/mps.sh stop || true
