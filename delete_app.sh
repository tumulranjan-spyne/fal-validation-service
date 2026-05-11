#!/usr/bin/env bash
# Delete the car-validation app from fal.ai.
#
# Usage:
#   ./delete_app.sh
#   FAL_TEAM=myteam ./delete_app.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate fal 2>/dev/null || true

APP_NAME="${APP_NAME:-car-validation}"

EXTRA=()
if [[ -n "${FAL_TEAM:-}" ]]; then
  EXTRA+=(--team "${FAL_TEAM}")
fi

echo "Deleting fal app: ${APP_NAME}"
set -- fal apps delete "${EXTRA[@]}" "${APP_NAME}" "$@"
exec "$@"