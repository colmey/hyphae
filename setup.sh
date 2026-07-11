#!/usr/bin/env bash
# setup.sh -- create the project virtualenv and install dependencies with uv.
#
# Uses `uv` (https://docs.astral.sh/uv/), which bundles its own Python and
# venv machinery, so it works even where the system `python3-venv`/`ensurepip`
# package is missing. If uv isn't installed, this script installs it.
#
# Usage:
#   ./setup.sh
#
# Idempotent: re-running reuses the existing .venv and re-syncs deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"

# --- Ensure uv is available -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found; installing it"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "error: need curl or wget to install uv. Install uv manually:" >&2
    echo "       https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
  # The installer drops uv in ~/.local/bin (or $XDG_BIN_HOME); make it visible.
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv version: $(uv --version)"

# --- Create the venv and install dependencies -------------------------------
echo "==> Creating virtualenv at $VENV"
uv venv --clear "$VENV"

echo "==> Installing dependencies from requirements.txt"
uv pip install --python "$VENV/bin/python" -r requirements.txt

# --- Seed .env --------------------------------------------------------------
if [ ! -f "$ROOT/.env" ]; then
  echo "==> Creating .env from .env.example"
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

echo
echo "Setup complete."
echo "  1. Edit .env and set your API keys (e.g. GEMINI_API_KEY) and MCP URLs."
echo "  2. Run the server:  ./runscript.sh -m uvicorn main:app --host 0.0.0.0 --port 8000"
echo "  3. Run hermetic tests: ./runscript.sh -m pytest"
echo "     Live integrations: ./runscript.sh -m pytest -m live"
