#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

count_done() {
  local directory="$1"
  if test -d "${directory}"; then
    find "${directory}" -type f -name DONE.json | wc -l
  else
    echo 0
  fi
}

if test -s results/runtime/autodl_stopline_chain.pid \
  && kill -0 "$(cat results/runtime/autodl_stopline_chain.pid)" 2>/dev/null; then
  echo "Stop-line chain: RUNNING"
else
  echo "Stop-line chain: NOT_RUNNING"
fi

for scenario in AR-S4 AR-S5 AR-S6 AR-S7; do
  warmup="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_warmup")"
  forks="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_fork")"
  root="results/phase1/SUPPORT_${scenario}_${scenario}_G2/Track-XAR"
  status="NOT_YET_RUN"
  test -f "${root}/test_metrics.json" && status="COMPLETED"
  echo "${scenario} screening: warmup=${warmup}/10 forks=${forks}/90 status=${status}"
done

for scenario in AR-S0 AR-S1 AR-S2 AR-S3; do
  warmup="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_warmup_critical")"
  forks="$(count_done "results/phase1/job_records/${scenario}_G2_XAR_fork_critical")"
  case "${scenario}" in
    AR-S0) root="results/phase1/E1_AR-S0_G2/Track-XAR" ;;
    AR-S1) root="results/phase1/E2_AR-S1_G2/Track-XAR" ;;
    AR-S2) root="results/phase1/E3_AR-S2_G2/Track-XAR" ;;
    AR-S3) root="results/phase1/E4_AR-S3_G2/Track-XAR" ;;
  esac
  status="NOT_YET_RUN"
  test -f "${root}/critical30_validation/validation_selection.json" \
    && status="COMPLETED"
  echo "${scenario} critical30 validation: added_warmup=${warmup}/20 added_forks=${forks}/180 status=${status}"
done

bootstrap=0
test -d results/phase1/E2_AR-S1_G2/M8/bootstrap \
  && bootstrap="$(find results/phase1/E2_AR-S1_G2/M8/bootstrap -type f -name 'seed_*.json' | wc -l)"
bootstrap_status="NOT_YET_RUN"
test -f results/phase1/E2_AR-S1_G2/M8/bootstrap_rank_audit.json \
  && bootstrap_status="COMPLETED"
echo "E2 M8 bootstrap: seeds=${bootstrap}/10 status=${bootstrap_status}"

if test -f results/runtime/stopline_compute_completed_at.txt; then
  echo "Stop-line compute: COMPLETED at $(cat results/runtime/stopline_compute_completed_at.txt)"
else
  echo "Stop-line compute: NOT_YET_RUN"
fi
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader

