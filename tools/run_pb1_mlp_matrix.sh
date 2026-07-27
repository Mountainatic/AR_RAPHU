#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(pwh|whpn)$ ]]; then
  echo "usage: $0 {pwh|whpn}" >&2
  exit 2
fi

dataset="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
candidate_root="$root/results/public_benchmarks/pb1/$dataset/development/MLPNARX_CHAMPNEYS2024/candidates"
mkdir -p "$candidate_root"
jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT

for width in 2 5 7 10; do
  for seed in 0 1 2 3 4; do
    candidate="$candidate_root/width_$(printf '%02d' "$width")_seed_$(printf '%02d' "$seed").json"
    if [[ ! -f "$candidate" ]]; then
      printf '%s %s\n' "$width" "$seed" >> "$jobs_file"
    fi
  done
done

if [[ ! -s "$jobs_file" ]]; then
  echo "all candidates already complete: $dataset"
  exit 0
fi

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export PYTHONPATH="$root/src"
export PB1_MLP_DATASET="$dataset"
export PB1_MLP_ROOT="$root"

xargs -a "$jobs_file" -n 2 -P 4 bash -c '
  width="$1"
  seed="$2"
  exec /root/AR_RAPHU_AUTODL/.venv/bin/python \
    "$PB1_MLP_ROOT/tools/run_pb1_mlp_narx.py" \
    --dataset "$PB1_MLP_DATASET" train \
    --width "$width" --seed "$seed"
' _
