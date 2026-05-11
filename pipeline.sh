#!/usr/bin/env bash
# Full pipeline: upload models -> deploy app.
#
# Usage:
#   export LOCAL_MODELS_DIR=/path/to/validation_models
#   ./pipeline.sh --yes                    # non-interactive deploy
#   ./pipeline.sh --verify-docker --yes    # build + smoke test, then upload + deploy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERIFY_DOCKER=0
DEPLOY_ARGS=()
LOCAL_MODEL_SRC="${LOCAL_MODELS_DIR:-validation_models}"
for arg in "$@"; do
  case "${arg}" in
    --verify-docker) VERIFY_DOCKER=1 ;;
    *) DEPLOY_ARGS+=("${arg}") ;;
  esac
done

if [[ "${VERIFY_DOCKER}" -eq 1 ]]; then
  echo "=== Building and smoke-testing Docker image ==="
  "${SCRIPT_DIR}/docker_smoke.sh"
fi

echo "=== Uploading models ==="
"${SCRIPT_DIR}/upload_models.sh" "${LOCAL_MODEL_SRC}"

echo "=== Deploying app ==="
LOCAL_MODELS_DIR="${LOCAL_MODEL_SRC}" "${SCRIPT_DIR}/deploy.sh" "${DEPLOY_ARGS[@]}"