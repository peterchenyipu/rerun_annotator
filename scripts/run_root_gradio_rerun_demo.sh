#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="${ROOT_DIR}/.venv/bin/python"
DEMO_PATH="${ROOT_DIR}/demo_gradio_rerun.py"

PORT="${GRADIO_SERVER_PORT:-17860}"
NUM_PORTS="${GRADIO_NUM_PORTS:-1}"

if [ ! -x "${PYTHON_PATH}" ]; then
  echo "error: expected Python interpreter at ${PYTHON_PATH}" >&2
  echo "hint: run 'uv sync' in ${ROOT_DIR} first" >&2
  exit 1
fi

echo "==> Launching root PyPI-based gradio_rerun demo on http://127.0.0.1:${PORT}"
(
  cd "${ROOT_DIR}"
  GRADIO_SERVER_PORT="${PORT}" GRADIO_NUM_PORTS="${NUM_PORTS}" "${PYTHON_PATH}" "${DEMO_PATH}"
)
