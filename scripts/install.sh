#!/usr/bin/env bash
# Render, install, enable, and start the Hyphae systemd service.
#
# Run as the normal account that should own the Hyphae process:
#   ./scripts/install.sh
set -euo pipefail

if [ "$EUID" -eq 0 ]; then
  echo "error: run this script as the normal Hyphae user, without sudo." >&2
  echo "       The script invokes sudo only for system service operations." >&2
  exit 1
fi

: "${USER:?USER must identify the account that will run Hyphae}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$ROOT/.deployment/hyphae.service"
UNIT_PATH="/etc/systemd/system/hyphae.service"

"$ROOT/scripts/setup.sh"

if grep -Eq '^(MCP_CONFIG_PATH|MODELS_CONFIG_PATH|ORCHESTRATOR_PROMPT_PATH|AGENT_PROMPT_PATH)=config/' "$ROOT/.env"; then
  echo "error: .env still uses configuration paths from the old repository layout." >&2
  echo "       Change config/... values to hyphae/config/... and rerun this script." >&2
  exit 1
fi

RENDERED_UNIT="$(mktemp --suffix=.service)"
cleanup() {
  rm -f "$RENDERED_UNIT"
}
trap cleanup EXIT

while IFS= read -r line || [ -n "$line" ]; do
  line="${line//\{USER\}/$USER}"
  line="${line//\{ROOT\}/$ROOT}"
  printf '%s\n' "$line"
done < "$TEMPLATE" > "$RENDERED_UNIT"

if grep -Eq '\{(USER|ROOT)\}' "$RENDERED_UNIT"; then
  echo "error: the rendered service contains an unresolved placeholder." >&2
  exit 1
fi

systemd-analyze verify "$RENDERED_UNIT"
sudo install -m 0644 "$RENDERED_UNIT" "$UNIT_PATH"
sudo systemctl daemon-reload
sudo systemctl enable --now hyphae
sudo systemctl status --no-pager hyphae

echo
echo "Hyphae is installed and running."
echo "Follow logs with: journalctl -u hyphae.service -f"
