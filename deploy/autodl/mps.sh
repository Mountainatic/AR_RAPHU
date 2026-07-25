#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
MPS_TAG="${AR_RAPHU_MPS_TAG:-ar_raphu_${UID}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/${MPS_TAG}_pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/${MPS_TAG}_log}"

require_control() {
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "nvidia-cuda-mps-control is unavailable; MPS is required." >&2
    exit 2
  fi
}

is_running() {
  test -s "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid" &&
    kill -0 "$(cat "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid")" 2>/dev/null
}

case "${ACTION}" in
  start)
    require_control
    install -d -m 700 "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if ! is_running; then
      nvidia-cuda-mps-control -d
    fi
    if ! is_running; then
      echo "MPS control daemon did not become healthy." >&2
      exit 3
    fi
    echo "MPS control daemon active: pipe=${CUDA_MPS_PIPE_DIRECTORY} log=${CUDA_MPS_LOG_DIRECTORY}"
    ;;
  status)
    require_control
    if ! is_running; then
      echo "MPS control daemon is not running." >&2
      exit 1
    fi
    echo "MPS control daemon active (pid $(cat "${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid"))."
    echo ps | nvidia-cuda-mps-control
    ;;
  stop)
    require_control
    if is_running; then
      echo quit | nvidia-cuda-mps-control
    fi
    echo "MPS control daemon stopped."
    ;;
  *)
    echo "Usage: $0 {start|status|stop}" >&2
    exit 64
    ;;
esac
