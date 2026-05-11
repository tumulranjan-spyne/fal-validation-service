#!/usr/bin/env bash
# Build the Docker image locally and smoke-test model loading.
# Does NOT require fal CLI — only Docker + NVIDIA Container Toolkit.
#
# Usage:
#   export LOCAL_MODELS_DIR=/path/to/validation_models
#   ./docker_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${CAR_VALIDATION_IMAGE:-car-validation:local}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running or you lack permission." >&2
  exit 1
fi

LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-validation_models}"

if [[ ! -d "${LOCAL_MODELS_DIR}" ]]; then
  echo "ERROR: Directory not found: ${LOCAL_MODELS_DIR}" >&2
  echo "Create it with model subdirectories, e.g.:" >&2
  echo "  ${LOCAL_MODELS_DIR}/auto-classification-roi/1/model.onnx" >&2
  exit 1
fi

echo "=== Building ${IMAGE} ==="
docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${SCRIPT_DIR}"

echo ""
echo "=== Smoke-testing model loading ==="
docker run --rm --gpus all \
  -v "$(pwd)/${LOCAL_MODELS_DIR}:/data/validation_models:ro" \
  -w /app "${IMAGE}" \
  python -c '
from model_loader import load_all_models
models = load_all_models()
print(f"Loaded {len(models)} models successfully!")
for k in sorted(models.keys()):
    print(f"  - {k}")
'

echo ""
echo "=== Smoke-testing inference ==="
docker run --rm --gpus all \
  -v "$(pwd)/${LOCAL_MODELS_DIR}:/data/validation_models:ro" \
  -w /app "${IMAGE}" \
  python -c '
import numpy as np
from model_loader import load_all_models
from preprocessing import preprocess_roi, preprocess_davit, preprocess_clip

models = load_all_models()
rgb = np.full((480, 640, 3), 127, dtype=np.uint8)

pre = preprocess_roi(rgb)
out = models["roi"].infer(pre)
print(f"ROI output shape: {np.array(out).shape}")

pre = preprocess_davit(rgb)
out = models["davit"].infer(pre[0])
print(f"Davit output shape: {np.array(out).shape}")

pre = preprocess_clip(rgb, center_crop=True)
emb = models["clip_backbone"].infer(pre)
out = models["general_classifier"].infer(emb)
print(f"General classifier output shape: {np.array(out).shape}")

print("All inference smoke tests passed!")
'