#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS=${PRISM_CPU_RESULTS:-/root/autodl-tmp/PRISM_CPU_RESULTS_V1_STRICT}
ATTEMPTS=0

while true; do
  if pgrep -f '[r]un_cpu_chain.py' >/dev/null; then
    sleep 30
    continue
  fi
  STATUS=$(${PRISM_PYTHON:-/root/AR_RAPHU_AUTODL/.venv/bin/python} -c \
    'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status","MISSING") if p.is_file() else "MISSING")' \
    "${RESULTS}/CHAIN_STATUS.json")
  if [[ "${STATUS}" == "COMPLETED" ]]; then
    exit 0
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  if (( ATTEMPTS > 8 )); then
    echo "WATCHDOG_STATUS=FAILED_RETRY_LIMIT" >&2
    exit 1
  fi
  PRISM_C3_JOBS=${PRISM_C3_JOBS:-2} bash "${PROJECT_DIR}/RUN_CPU_CHAIN.sh" >> "${PROJECT_DIR}/logs/cpu_chain_stdout.log" 2>&1 || true
done
