#!/usr/bin/env bash
# Install Python dependencies into the fal conda environment.
#
# Usage:
#   ./bootstrap_local_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate fal 2>/dev/null || {
  echo "ERROR: No 'fal' conda environment found. Create it first:"
  echo "  conda create -n fal python=3.11"
  echo "  conda activate fal"
  echo "  pip install fal"
  exit 1
}

REQ="${SCRIPT_DIR}/requirements.txt"
echo "Installing from ${REQ} (this may take a while due to torch/onnxruntime)..."
pip install -r "${REQ}" --extra-index-url https://download.pytorch.org/whl/cu124
echo "Done. Now you can run:"
echo "  ./docker_smoke.sh              # local smoke test"
echo "  ./upload_models.sh             # upload weights to fal"
echo "  ./deploy.sh                    # deploy to fal.ai"