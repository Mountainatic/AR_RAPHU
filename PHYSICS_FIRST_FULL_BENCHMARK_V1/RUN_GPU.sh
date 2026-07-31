#!/usr/bin/env bash
set -u -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED=""
CPU_RESULTS=""
DEVICE="cuda:0"
SCREENING_SEEDS="0,1,2,3,4"
FINAL_SEEDS="0,1,2,3,4,5,6,7,8,9"
RESULTS="$ROOT/results_gpu"
RUN_SMOKE=1
RUN_CORE=1
RUN_FRONTIER=1
RUN_FINALISTS=0
STRICT_FOLDS=1

usage() {
  cat <<'EOF'
Usage: bash RUN_GPU.sh --shared PATH [options]
  --cpu-results PATH       CPU results root; enables K-residual ablations
  --device cuda:0
  --screening-seeds 0,1,2,3,4
  --final-seeds 0,1,2,3,4,5,6,7,8,9
  --results PATH
  --skip-smoke | --skip-core | --skip-frontier
  --run-finalists
  --no-strict-folds        only for debugging, not formal results
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared) SHARED="$2"; shift 2 ;;
    --cpu-results) CPU_RESULTS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --screening-seeds) SCREENING_SEEDS="$2"; shift 2 ;;
    --final-seeds) FINAL_SEEDS="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --skip-smoke) RUN_SMOKE=0; shift ;;
    --skip-core) RUN_CORE=0; shift ;;
    --skip-frontier) RUN_FRONTIER=0; shift ;;
    --run-finalists) RUN_FINALISTS=1; shift ;;
    --no-strict-folds) STRICT_FOLDS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SHARED" ]]; then
  echo "--shared is required" >&2
  usage
  exit 2
fi

if [[ -d "$ROOT/.venv-gpu" ]]; then
  source "$ROOT/.venv-gpu/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RESULTS/logs"

STRICT_ARG="--strict-folds"
if [[ "$STRICT_FOLDS" -eq 0 ]]; then STRICT_ARG="--no-strict-folds"; fi
CPU_ARG=()
CORE_MODELS="mlp_u,lstm_u,gru_u,tcn_u,dlinear_u,nlinear_u,lstm_sa_u,lstm_uxy,gru_uxy,tcn_uxy"
FRONTIER_MODELS="transformer_u,transformer_uxy,patchtst_u,patchtst_uxy,timesnet_u,timesnet_uxy,static_gcn_u,t_akgnn_u,t_akgnn_uxy,adaptive_graph_kan_u,no_graph_kan_u,gru_vae_u,gru_vae_uxy"
if [[ -n "$CPU_RESULTS" ]]; then
  CPU_ARG=(--cpu-results "$CPU_RESULTS")
  CORE_MODELS="$CORE_MODELS,k_residual_gru,k_residual_tcn"
  FRONTIER_MODELS="$FRONTIER_MODELS,k_residual_transformer"
fi

status=0
python "$ROOT/scripts/validate_shared_dataset.py" --shared "$SHARED" --output "$RESULTS/SHARED_VALIDATION.json" || exit $?
python -m pytest -q "$ROOT/tests" || exit $?

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  python "$ROOT/scripts/run_gpu_smoke.py" \
    --shared "$SHARED" --results "$RESULTS" --device "$DEVICE" \
    --seeds 0 --epochs 2 --no-strict-folds \
    "${CPU_ARG[@]}" 2>&1 | tee "$RESULTS/logs/smoke.log"
  rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || status=$rc
fi

if [[ "$RUN_CORE" -eq 1 ]]; then
  python "$ROOT/scripts/run_gpu_stage1_core.py" \
    --shared "$SHARED" --results "$RESULTS" --device "$DEVICE" \
    --models "$CORE_MODELS" --seeds "$SCREENING_SEEDS" "$STRICT_ARG" \
    "${CPU_ARG[@]}" 2>&1 | tee "$RESULTS/logs/core.log"
  rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || status=$rc
fi

if [[ "$RUN_FRONTIER" -eq 1 ]]; then
  python "$ROOT/scripts/run_gpu_stage2_frontier.py" \
    --shared "$SHARED" --results "$RESULTS" --device "$DEVICE" \
    --models "$FRONTIER_MODELS" --seeds "$SCREENING_SEEDS" "$STRICT_ARG" \
    "${CPU_ARG[@]}" 2>&1 | tee "$RESULTS/logs/frontier.log"
  rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || status=$rc
fi

if [[ "$RUN_FINALISTS" -eq 1 ]]; then
  python "$ROOT/scripts/run_gpu_stage3_finalists.py" \
    --shared "$SHARED" --results "$RESULTS" --device "$DEVICE" \
    --seeds "$FINAL_SEEDS" --strict-folds \
    "${CPU_ARG[@]}" 2>&1 | tee "$RESULTS/logs/finalists.log"
  rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || status=$rc
fi

python "$ROOT/scripts/aggregate_gpu_results.py" --results "$RESULTS"
rm -rf "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS"
rm -f "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS_bundle.zip" "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS_bundle.zip.sha256"
python "$ROOT/scripts/build_gpu_return_bundle.py" \
  --source-root "$ROOT" --results "$RESULTS" \
  --output-dir "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS" \
  --keep-best-checkpoints-only
python "$ROOT/scripts/validate_package.py" --package-dir "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS"
python "$ROOT/scripts/validate_zip_roundtrip.py" --zip "$ROOT/return/PHYSICS_FIRST_GPU_RESULTS_bundle.zip"

exit "$status"
