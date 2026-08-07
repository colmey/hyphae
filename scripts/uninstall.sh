#!/usr/bin/env bash
# Stop, disable, and remove the Hyphae systemd service.
#
# Run as the normal account used to install the service:
#   ./scripts/uninstall.sh
set -euo pipefail

if [ "$EUID" -eq 0 ]; then
  echo "error: run this script as a normal user, without sudo." >&2
  echo "       The script invokes sudo only for system service operations." >&2
  exit 1
fi

UNIT_PATH="/etc/systemd/system/hyphae.service"

if [ ! -e "$UNIT_PATH" ]; then
  echo "Hyphae is not installed at $UNIT_PATH."
  exit 0
fi

sudo systemctl disable --now hyphae
sudo rm "$UNIT_PATH"
sudo systemctl daemon-reload
sudo systemctl reset-failed hyphae 2>/dev/null || true

echo "Hyphae's systemd service was removed. Repository files and data were retained."
