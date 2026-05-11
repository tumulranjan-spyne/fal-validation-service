#!/usr/bin/env bash
# Deploy CarValidationApp to fal.ai serverless.
#
# Prerequisites:
#   conda env named 'fal' with `fal` CLI installed and authenticated.
#   Models uploaded via ./upload_models.sh
#
# Usage:
#   ./deploy.sh
#   ./deploy.sh --check --yes        # non-interactive, show plan first
#   FAL_TEAM=myteam ./deploy.sh      # deploy to a specific team
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate fal conda env
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate fal 2>/dev/null || {
  command -v fal &>/dev/null || {
    echo "ERROR: 'fal' CLI not found. Install with: pip install fal" >&2
    exit 1
  }
}

cd "${SCRIPT_DIR}"

# Ensure models are uploaded first
echo "=== Checking model files exist locally ==="
MODEL_SRC="${LOCAL_MODELS_DIR:-validation_models}"
if [[ ! -d "${MODEL_SRC}" ]]; then
  echo "WARNING: Local model directory not found: ${MODEL_SRC}"
  echo "Run ./upload_models.sh first to upload models to fal storage."
  echo ""
fi

# Locate fal binary reliably
FAL_BIN=""
for candidate in \
  "/home/spyne-4090/miniconda3/envs/fal/bin/fal" \
  "$(which fal 2>/dev/null)" \
  "$(conda run -n fal which fal 2>/dev/null)"; do
  if [[ -x "${candidate}" ]]; then
    FAL_BIN="${candidate}"
    break
  fi
done

if [[ -z "${FAL_BIN}" ]]; then
  echo "ERROR: 'fal' CLI not found." >&2
  exit 1
fi

FAL_TEAM="${FAL_TEAM:-Spyne-AI}"

EXTRA=(--team "${FAL_TEAM}")

set -- "${FAL_BIN}" deploy car-validation "${EXTRA[@]}" "$@"
echo "Running: $*"
exec "$@"