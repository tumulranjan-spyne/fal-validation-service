#!/usr/bin/env bash
# Upload model weights to fal persistent storage under /data/validation_models/.
#
# Directory layout expected locally:
#   validation_models/
#   ├── auto-classification-roi/1/model.onnx
#   ├── auto-classification-roi-davit/1/model.onnx
#   ├── auto-classification-angle_classification/1/model.onnx
#   ├── cgi_classifier/1/model.onnx
#   ├── haze_classifier/1/model.onnx
#   ├── clip_vit_B32/1/model.pt
#   ├── general_classifier/1/model.pt
#   ├── car_type_classifier_clip/1/model.pt
#   ├── auto-segmentation-exterior_removebg/1/model.pt
#   ├── focus_classifier/1/model.pt
#   ├── resnet_backbone_clf/1/model.pt
#   ├── resnet_backbone_clf/2/model.pt
#   └── screen_classification/1/model.onnx   (optional)
#
# Prerequisites:
#   conda env named 'fal' with `fal` CLI installed and authenticated.
#
# Usage:
#   ./upload_models.sh                         # defaults to ./validation_models
#   ./upload_models.sh /path/to/validation_models
#   FAL_TEAM=myteam ./upload_models.sh         # upload to a specific team's storage
set -euo pipefail

LOCAL_MODELS_DIR="${1:-validation_models}"
FAL_DST_PREFIX="validation_models"
FAL_TEAM="${FAL_TEAM:-Spyne-AI}"

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
  echo "ERROR: 'fal' CLI not found. Install with: pip install fal" >&2
  echo "  Tried: /home/spyne-4090/miniconda3/envs/fal/bin/fal, PATH, conda run" >&2
  exit 1
fi

if [[ ! -d "${LOCAL_MODELS_DIR}" ]]; then
  echo "ERROR: Directory not found: ${LOCAL_MODELS_DIR}" >&2
  echo "Create it or pass the path as the first argument." >&2
  exit 1
fi

# Model list: fal-storage-path -> local relative path under LOCAL_MODELS_DIR
declare -a MODELS=(
    "auto-classification-roi/1/model.onnx"
    "auto-classification-roi-davit/1/model.onnx"
    "auto-classification-angle_classification/1/model.onnx"
    "cgi_classifier/1/model.onnx"
    "haze_classifier/1/model.onnx"
    "clip_vit_B32/1/model.pt"
    "general_classifier/1/model.pt"
    "car_type_classifier_clip/1/model.pt"
    "auto-segmentation-exterior_removebg/1/model.pt"
    "focus_classifier/1/model.pt"
    "resnet_backbone_clf/1/model.pt"
    "resnet_backbone_clf/2/model.pt"
)

# Optional models (won't fail if missing)
declare -a OPTIONAL_MODELS=(
    "screen_classification/1/model.onnx"
)

EXTRA=(--team "${FAL_TEAM}")

echo "Uploading models from ${LOCAL_MODELS_DIR} to fal://${FAL_DST_PREFIX}/ (team: ${FAL_TEAM}) ..."
echo "Using fal binary: ${FAL_BIN}"
echo ""

uploaded=0
for rel in "${MODELS[@]}"; do
    src="${LOCAL_MODELS_DIR}/${rel}"
    dst="${FAL_DST_PREFIX}/${rel}"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: Required model missing: $src"
        exit 1
    fi
    echo "[upload] $src -> ${dst}"
    if output=$("${FAL_BIN}" files upload "${EXTRA[@]}" "$src" "$dst" 2>&1); then
        echo "$output"
        uploaded=$((uploaded + 1))
    elif [[ "$output" == *"already exists"* ]]; then
        echo "$output"
        echo "[info] File already exists, continuing..."
        uploaded=$((uploaded + 1))
    else
        echo "ERROR: Failed to upload $src"
        echo "$output"
        exit 1
    fi
done

for rel in "${OPTIONAL_MODELS[@]}"; do
    src="${LOCAL_MODELS_DIR}/${rel}"
    dst="${FAL_DST_PREFIX}/${rel}"
    if [[ ! -f "$src" ]]; then
        echo "[upload] Skipping optional: $src (not found)"
        continue
    fi
    echo "[upload] $src -> ${dst}"
    if output=$("${FAL_BIN}" files upload "${EXTRA[@]}" "$src" "$dst" 2>&1); then
        echo "$output"
        uploaded=$((uploaded + 1))
    elif [[ "$output" == *"already exists"* ]]; then
        echo "$output"
        echo "[info] File already exists, continuing..."
        uploaded=$((uploaded + 1))
    else
        echo "ERROR: Failed to upload $src"
        echo "$output"
        exit 1
    fi
done

echo ""
echo "Done. Uploaded ${uploaded} models."
echo "Verify: fal files list data/${FAL_DST_PREFIX}/"