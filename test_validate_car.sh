#!/usr/bin/env bash
# Test the deployed /validate_car endpoint with a sample image.
#
# Required env:
#   FAL_APP_URL   — e.g. https://fal.run/<workspace>/car-validation
#   TEST_IMAGE_URL — public image URL
# Optional:
#   FAL_KEY       — API key if app is private
#
# Usage:
#   export FAL_APP_URL="https://fal.run/myworkspace/car-validation"
#   export TEST_IMAGE_URL="https://example.com/car.jpg"
#   ./test_validate_car.sh
set -euo pipefail

: "${FAL_APP_URL:?Set FAL_APP_URL to your deployed app URL}"
: "${TEST_IMAGE_URL:?Set TEST_IMAGE_URL to a public image URL}"

URL="${FAL_APP_URL%/}/validate_car"

BODY=$(python3 -c '
import json, os
body = {
    "image_url": os.environ["TEST_IMAGE_URL"],
    "car_cls": True,
    "car_shoot_category_cls": True,
    "car_type_cls": True,
    "angle_detect": True,
    "haze_classification": True,
}
print(json.dumps(body, indent=2))
')

HDR=(-H "Content-Type: application/json")
if [[ -n "${FAL_KEY:-}" ]]; then
  HDR+=(-H "Authorization: Key ${FAL_KEY}")
fi

echo "POST ${URL}"
echo "---"
RESP=$(curl -sS -X POST "${HDR[@]}" -d "${BODY}" "${URL}")
RC=$?

if [[ $RC -ne 0 ]]; then
  echo "ERROR: curl exited with code $RC"
  echo "${RESP}"
  exit 1
fi

if command -v jq &>/dev/null; then
  echo "${RESP}" | jq .
else
  echo "${RESP}"
fi