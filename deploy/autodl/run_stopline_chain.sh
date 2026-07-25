#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs results/runtime

extension_pid_file="results/runtime/autodl_phase1_extensions.pid"
if test -s "${extension_pid_file}"; then
  extension_pid="$(cat "${extension_pid_file}")"
  while kill -0 "${extension_pid}" 2>/dev/null; do
    sleep 30
  done
fi

for scenario in AR-S4 AR-S5 AR-S6 AR-S7; do
  result="results/phase1/SUPPORT_${scenario}_${scenario}_G2/Track-XAR/test_metrics.json"
  if ! test -f "${result}"; then
    echo "Missing prerequisite extension result: ${result}" >&2
    exit 65
  fi
done

bash deploy/autodl/run_phase1_critical30_validation.sh
bash deploy/autodl/run_phase1_m8_bootstrap.sh

date -u +%Y-%m-%dT%H:%M:%SZ > results/runtime/stopline_compute_completed_at.txt
echo "Stop-line compute chain completed."
