#!/usr/bin/env bash
# Run CarValidationApp locally (same Docker image, local GPU).
# Useful for debugging before deploying to fal.ai.
#
# Usage:
#   ./run_local.sh
#   FAL_TEAM=myteam ./run_local.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate fal 2>/dev/null || true

cd "${SCRIPT_DIR}"

# Mount local models into the container at the expected path
LOCAL_MODEL_SRC="${LOCAL_MODELS_DIR:-validation_models}"

if [[ ! -d "${LOCAL_MODEL_SRC}" ]]; then
  echo "ERROR: Local model directory not found: ${LOCAL_MODEL_SRC}" >&2
  echo "Set LOCAL_MODELS_DIR or create the validation_models/ directory." >&2
  exit 1
fi

MOUNT_OPTS="--volume $(pwd)/${LOCAL_MODEL_SRC}:/data/validation_models:ro"

FAL_TEAM="${FAL_TEAM:-Spyne-AI}"
EXTRA=(--local --team "${FAL_TEAM}")

APP_REF="${APP_REF:-app.py::CarValidationApp}"

set -- fal run "${EXTRA[@]}" "${MOUNT_OPTS}" "${APP_REF}" "$@"
echo "Running: $*"
exec "$@"