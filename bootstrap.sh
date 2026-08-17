#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"

"${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install \
  "matplotlib==3.11.1" \
  "numpy==2.3.5" \
  "pillow==12.0.0" \
  "modelscope-hub==0.2.0"

cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" -m compileall -q scripts tests
echo "Analysis environment ready: ${VENV_DIR}"
