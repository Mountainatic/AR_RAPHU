#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/.autodl-tools"
UV_BIN="${TOOLS_DIR}/uv"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif test ! -x "${UV_BIN}"; then
  mkdir -p "${TOOLS_DIR}"
  INSTALLER="${TOOLS_DIR}/uv-install.sh"
  curl --fail --location --silent --show-error \
    https://astral.sh/uv/install.sh --output "${INSTALLER}"
  UV_INSTALL_DIR="${TOOLS_DIR}" sh "${INSTALLER}"
fi

if test ! -x "${UV_BIN}"; then
  echo "uv installation failed: ${UV_BIN}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv"
"${UV_BIN}" sync --frozen

echo "uv environment ready: $("${UV_BIN}" --version)"
echo "Next: bash deploy/autodl/verify_server.sh"
