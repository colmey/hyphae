#!/usr/bin/env bash
# runscript.sh -- project-environment Python launcher.
#
# Activates the project virtualenv, prepends the project root to PYTHONPATH
# (so scripts under tests/ can import config, main, etc.), and
# runs Python with whatever arguments you pass.
#
# Examples:
#   ./runscript.sh -m pytest tests/test_config.py
#   ./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
  echo "error: virtualenv not found at $VENV — run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python "$@"
