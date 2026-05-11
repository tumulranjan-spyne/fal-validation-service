"""
Car Validation App for fal.ai serverless GPU inference.

Deploy:  fal deploy car-validation
Test:    curl -X POST https://fal.run/<workspace>/car-validation/validate_car \
           -H "Content-Type: application/json" \
           -d '{"image_url":"https://...", "car_cls":true, "car_type_cls":true}'
"""

import os
import time
import base64
import urllib.request
import traceback
from typing import Optional, Dict, Any, Tuple

import fal
from fal.container import ContainerImage
from pydantic import BaseModel, ConfigDict


def _np_cv_pp():
    """Lazy import for numpy/cv2/preprocessing so isolate can unpickle app.py without them."""
    import numpy as np
    import cv2
    import preprocessing as pp

    return np, cv2, pp


# ============================================================================
# ROI / Davit label maps (from imagewizard src/solutions/exterior/classification/roi/)
# ============================================================================
COARSE_ROI_LABELS = ["focus", "outer", "inside", "misc"]

DAVIT_LABELS = [
    "Bonnet", "Door_Handle", "Headlight", "Roof", "Tyre", "Tyre_tread",
    "Bottom", "Engine", "Focus_Shoot", "Interior", "Interior_Focus_Shoot",
    "key", "marketing", "Open_Door", "open_door_wide", "Reg", "Side_mirror",
    "text", "Trunk",
]

ROI_SUBROI_MAP = {
    "Bottom": "inside", "Door_Handle": "focus", "Engine": "inside",
    "Focus_Shoot": "focus", "Headlight": "focus", "Interior": "inside",
    "Interior_Focus_Shoot": "inside", "key": "misc", "Open_Door": "inside",
    "open_door_wide": "outer", "Reg": "inside", "Side_mirror": "focus",
    "Trunk": "inside", "Tyre": "focus", "Tyre_tread": "focus",
    "Bonnet": "misc", "Roof": "misc", "marketing": "misc", "text": "misc",
}

CAR_ROI_DISPLAY = {
    "outer": "Exterior", "inside": "Interior",
    "misc": "Miscellaneous", "focus": "Focus", "interior_360": "360int",
}

ANGLE_LABELS = [
    0, 10, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310,
    320, 330, 340, 350, 40, 50, 60, 70, 80, 90,
]

GENERAL_LABELS = ['Human', 'Jewellery', 'Automobile', 'Footwear', 'Bag', 'Food', 'Food']
CAR_TYPE_LABELS = ["Hatchback", "SUV", "Sedan"]


# ============================================================================
# Request / Response schemas
# ============================================================================
class CarValidationInput(BaseModel):
    """POST body: flat JSON. image_url required; booleans default false.

    422 Unprocessable Entity = schema mismatch; see HTTP response ``detail`` or
    fal logs for which ``loc`` / ``msg`` failed (often missing image_url, wrong
    types, or extra keys if client sends unsupported fields).
    """

    model_config = ConfigDict(extra="ignore")

    image_url: str
    car_cls: bool = False
    car_shoot_category_cls: bool = False
    car_type_cls: bool = False
    car_inter_interior_cls: bool = False
    sub_cat_cls: bool = False
    number_plate_detection: bool = False
    angle_detect: bool = False
    required_angle: Optional[int] = None
    crop_detect: bool = False
    distance_detect: bool = False
    exposure_detect: bool = False
    reflection_detect: bool = False
    tyre_mud_detect: bool = False
    tilt_detect: bool = False
    screen_detect: bool = False
    haze_classification: bool = False
    check_cgi: bool = False
    additional_data: str = ""


class CarValidationOutput(BaseModel):
    is_car: Optional[Dict[str, Any]] = None
    car_shoot_category: Optional[Dict[str, Any]] = None
    car_type: Optional[Dict[str, Any]] = None
    interior_class: Optional[Dict[str, Any]] = None
    window_tint: Optional[Dict[str, Any]] = None
    number_plate: Optional[Dict[str, Any]] = None
    angle: Optional[Dict[str, Any]] = None
    crop_array: Optional[Dict[str, Any]] = None
    distance: Optional[Dict[str, Any]] = None
    exposure: Optional[Dict[str, Any]] = None
    reflection: Optional[Dict[str, Any]] = None
    tyre_mud: Optional[Dict[str, Any]] = None
    tilt_value: Optional[Dict[str, Any]] = None
    sub_category: Optional[Dict[str, Any]] = None
    is_haze: Optional[bool] = None
    check_cgi: Optional[Dict[str, Any]] = None
    screen_detect: Optional[Dict[str, Any]] = None
    additional_data: str = ""


# ============================================================================
# Main fal.App
# ============================================================================
class CarValidationApp(fal.App):
    # fal runs lifespan() + setup() in the isolate interpreter. Your Dockerfile installs
    # these for the GPU image layers, but the agent still needs the same wheels here.
    # See fal Host._CONTAINER_KEYS: image and requirements are merged for kind="container".
    requirements = [
        [
            "--index-url",
            "https://download.pytorch.org/whl/cu129",
            "torch==2.8.0",
            "torchvision==0.23.0",
        ],
        [
            "onnxruntime-gpu==1.19.2",
            "opencv-python-headless==4.10.0.84",
            "pillow==10.4.0",
            "numpy==1.26.4",
        ],
    ]

    # Isolate deserializes this module in a bare Python env (before requirements install).
    data_mounts = ["/data"]

    machine_type = ["GPU-RTX5090"]
    image = ContainerImage.from_dockerfile("Dockerfile")
    keep_alive = 300
    min_concurrency = 1
    max_concurrency = 8
    scaling_delay = 120
    # Many ONNX sessions + TorchScript on cold GPU; default 600s often kills mid-startup.
    startup_timeout = 3600
    request_timeout = 3600

    def setup(self) -> None:
        import model_loader as ml

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print("[setup] CarValidationApp setup() — loading models ...")
        t0 = time.perf_counter()
        try:
            print(f"[setup] cwd={os.getcwd()!r} ml.DATA_DIR={ml.DATA_DIR!r}")
            print(
                f"[setup] /data exists={os.path.isdir('/data')} "
                f"ismount={os.path.ismount('/data')}"
            )
            if os.path.isdir("/data"):
                top = sorted(os.listdir("/data"))
                print(f"[setup] /data entries ({len(top)}): {top[:40]!r}{' ...' if len(top) > 40 else ''}")
            if os.path.isdir(ml.DATA_DIR):
                sub = os.listdir(ml.DATA_DIR)
                print(f"[setup] {ml.DATA_DIR!r} entries ({len(sub)}): {sub[:40]!r}{' ...' if len(sub) > 40 else ''}")
            missing = [p for p in ml.list_required_weight_paths() if not os.path.isfile(p)]
            if missing:
                print(
                    f"[setup] required files still missing ({len(missing)}); "
                    f"first few: {[os.path.relpath(p, ml.DATA_DIR) for p in missing[:5]]}"
                )
            self.models = ml.load_all_models()
            print(f"[setup] ready in {time.perf_counter() - t0:.1f}s")
        except BaseException:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Image download helpers
    # ------------------------------------------------------------------
    def _download_rgb(self, url: str) -> Any:
        np, cv2, _pp = _np_cv_pp()
        req = urllib.request.Request(url, headers={"User-Agent": "fal-car-validation/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot decode image: {url[:120]}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Softmax helper
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x: Any) -> Any:
        import numpy as np

        x = x.astype(np.float64)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # ------------------------------------------------------------------
    # ROI classification (mirrors imagewizard classify_roi + classify_roi_2)
    # ------------------------------------------------------------------
    def _classify_roi(self, img_rgb: Any) -> Tuple[str, Optional[str], float]:
        """Returns (roi_label, subroi_label, confidence)."""
        np, _, pp = _np_cv_pp()
        pre = pp.preprocess_roi(img_rgb)
        logits = np.squeeze(self.models["roi"].infer(pre)).astype(np.float32)
        idx = int(np.argmax(logits))
        coarse = COARSE_ROI_LABELS[idx]
        conf = float(self._softmax(logits)[idx])

        if coarse == "outer":
            return coarse, None, conf

        # Davit fine-grained
        davit_in = pp.preprocess_davit(img_rgb)[0]
        raw = np.squeeze(self.models["davit"].infer(davit_in)).astype(np.float32)
        if raw.ndim != 1:
            raw = raw.reshape(-1)
        top = int(np.argmax(raw))
        label = DAVIT_LABELS[top]
        davit_conf = float(self._softmax(raw)[top])

        # Tyre/Focus_Shoot ambiguity
        if label == "Tyre" and coarse == "focus" and davit_conf < 0.8:
            label = "Focus_Shoot"

        roi = ROI_SUBROI_MAP.get(label, "misc")
        if roi == "outer":
            return roi, None, davit_conf
        return roi, label, davit_conf

    # ------------------------------------------------------------------
    # General classifier (CLIP backbone -> general_classifier head)
    # ------------------------------------------------------------------
    def _general_classify(self, img_rgb: Any) -> Tuple[str, float]:
        np, _, pp = _np_cv_pp()
        pre = pp.preprocess_clip(img_rgb, center_crop=True)
        emb = self.models["clip_backbone"].infer(pre)
        logits = np.squeeze(self.models["general_classifier"].infer(emb)).astype(np.float64)
        idx = int(np.argmax(logits))
        conf = float(self._softmax(logits)[idx]) * 100.0
        return GENERAL_LABELS[idx], conf

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @fal.endpoint("/health")
    def health(self) -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Unified car validation endpoint
    # ------------------------------------------------------------------
    # Parameter must not be named "inp" (fal gateway) or "body" (also routed as query).
    @fal.endpoint("/validate_car")
    def validate_car(self, payload: CarValidationInput) -> CarValidationOutput:
        t0 = time.perf_counter()

        # Default result template (matches imagewizard CarDetection.result_template)
        result: Dict[str, Any] = {
            "is_car": None,
            "car_shoot_category": {"value": "Not a car image."},
            "car_type": {"value": "Not an exterior car image."},
            "interior_class": {"value": "Not an interior car image."},
            "window_tint": {"value": "Not an exterior car image."},
            "number_plate": {"value": "Not an exterior car image."},
            "angle": {"value": "Not an exterior car image."},
            "crop_array": {"value": "Not an exterior car image."},
            "distance": {"value": "Not an exterior car image."},
            "exposure": {"value": "Not an exterior car image."},
            "reflection": {"value": "Not an exterior car image."},
            "tyre_mud": {"value": "Not an exterior car image."},
            "tilt_value": {"value": "Not an exterior car image."},
            "sub_category": {"value": "Not an interior/misc car image."},
            "is_haze": None,
            "check_cgi": None,
            "screen_detect": None,
            "additional_data": payload.additional_data,
        }

        try:
            np, cv2, pp = _np_cv_pp()
            img_rgb = self._download_rgb(payload.image_url)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # ---- Step 1: General classification ----
            if payload.car_cls:
                cat, conf = self._general_classify(img_rgb)
                result["is_car"] = {
                    "value": cat == "Automobile",
                    "confidence": round(conf, 2),
                }

            # ---- Step 2: ROI classification ----
            tolerance = 0.01
            h_img, w_img = img_rgb.shape[:2]
            aspect_ratio = w_img / h_img

            if abs(aspect_ratio - 2) <= tolerance:
                raw_roi = "interior_360"
                roi_conf = 1.0
            else:
                raw_roi, _, roi_conf = self._classify_roi(img_rgb)

            roi_display = CAR_ROI_DISPLAY.get(raw_roi, raw_roi)

            if payload.car_shoot_category_cls:
                result["car_shoot_category"] = {
                    "value": roi_display,
                    "confidence": round(abs(roi_conf), 2),
                }
            else:
                result["car_shoot_category"] = None

            # ---- Step 3: Interior / Focus sub-classification ----
            if payload.car_inter_interior_cls or payload.sub_cat_cls:
                if roi_display == "Interior":
                    _, sub_label, sub_conf = self._classify_roi(img_rgb)
                    result["interior_class"] = {
                        "value": sub_label,
                        "confidence": round(abs(sub_conf), 2),
                    }
                    result["sub_category"] = dict(result["interior_class"])
                elif roi_display == "Focus":
                    _, sub_label, sub_conf = self._classify_roi(img_rgb)
                    result["sub_category"] = {
                        "value": sub_label,
                        "confidence": round(abs(sub_conf), 2),
                    }
                else:
                    result["interior_class"] = None
                    result["sub_category"] = None
            else:
                result["interior_class"] = None
                result["sub_category"] = None

            # ---- Step 4: Exterior-only analyses ----
            if roi_display == "Exterior":
                result["number_plate"] = None

                # Segment car
                seg_in = pp.preprocess_segmentation(img_bgr)
                mask_arr = self.models["segmentation"].infer(seg_in[0])
                mask = mask_arr.squeeze().astype(np.uint8)
                if mask.shape != img_bgr.shape[:2]:
                    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                mask = (mask > 128).astype(np.uint8) * 255
                tp_rgba = np.dstack([img_rgb, mask])

                # Car type
                if payload.car_type_cls:
                    type_pre = pp.preprocess_car_type(img_rgb, mask)
                    type_logits = self.models["car_type"].infer(type_pre)
                    type_idx = int(np.argmax(type_logits))
                    type_conf = float(self._softmax(type_logits)[type_idx]) * 100
                    result["car_type"] = {
                        "value": CAR_TYPE_LABELS[type_idx],
                        "confidence": round(type_conf, 2),
                    }

                # Angle
                if payload.angle_detect:
                    angle_pre = pp.preprocess_angle(tp_rgba)
                    angle_logits = self.models["angle"].infer(angle_pre)
                    angle_idx = int(np.argmax(angle_logits))
                    angle_conf = float(self._softmax(angle_logits)[angle_idx]) * 100
                    angle_val = ANGLE_LABELS[angle_idx]
                    result["angle"] = {
                        "value": angle_val,
                        "confidence": round(angle_conf, 2),
                    }
                    if payload.required_angle is not None:
                        result["angle"]["accepted"] = False
                        offset = abs(payload.required_angle - angle_val)
                        if offset <= 10 or offset >= 350:
                            result["angle"]["accepted"] = True

                # Crop detection
                if payload.crop_detect:
                    alpha = tp_rgba[:, :, 3]
                    ys, xs = np.nonzero(alpha > 127)
                    if len(xs) > 0:
                        crop = {"left": False, "top": False, "right": False, "bottom": False}
                        hh, ww = tp_rgba.shape[:2]
                        if xs.min() < 3: crop["left"] = True
                        if ys.min() < 3: crop["top"] = True
                        if xs.max() >= ww - 3: crop["right"] = True
                        if ys.max() >= hh - 3: crop["bottom"] = True
                        result["crop_array"] = {"value": crop}

                # Distance
                if payload.distance_detect:
                    ys, xs = np.nonzero(mask > 0)
                    if len(xs) > 0:
                        ratio = (ys.max() - ys.min()) / h_img
                        result["distance"] = {"value": "bad" if ratio < 0.4 else "good"}

                # Exposure
                if payload.exposure_detect:
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    _, _, v = cv2.split(hsv)
                    mean_v = float(np.mean(v[np.where(v != 255)])) if np.any(v != 255) else 128.0
                    if mean_v < 90:
                        exp = "Low"
                    elif mean_v < 180:
                        exp = "Med"
                    else:
                        exp = "High"
                    result["exposure"] = {"value": exp}

                # Reflection
                if payload.reflection_detect:
                    refl_pre = pp.preprocess_reflection_tp(tp_rgba)
                    refl_logits = self.models["resnet_v1"].infer(refl_pre)
                    refl_idx = int(np.argmax(refl_logits))
                    refl_conf = float(self._softmax(refl_logits)[refl_idx]) * 100
                    result["reflection"] = {
                        "value": "high" if refl_idx == 1 else "low",
                        "confidence": round(refl_conf, 2),
                    }

                # Tyre mud
                if payload.tyre_mud_detect:
                    mud_pre = pp.preprocess_224_pil(img_rgb)
                    mud_logits = self.models["resnet_v2"].infer(mud_pre)
                    mud_idx = int(np.argmax(mud_logits))
                    mud_conf = float(self._softmax(mud_logits)[mud_idx]) * 100
                    result["tyre_mud"] = {
                        "value": "bad" if mud_idx == 1 else "good",
                        "confidence": round(mud_conf, 2),
                    }

                # Tilt (placeholder — no tilt model in this deployment)
                if payload.tilt_detect and payload.angle_detect:
                    result["tilt_value"] = {"value": 0.0}

            # ---- Step 5: ROI-independent checks ----
            if payload.screen_detect:
                if "screen_classifier" in self.models:
                    screen_pre = pp.preprocess_224_pil(img_rgb)
                    screen_logits = self.models["screen_classifier"].infer(screen_pre)
                    screen_idx = int(np.argmax(screen_logits))
                    screen_conf = float(self._softmax(screen_logits)[screen_idx]) * 100
                    result["screen_detect"] = {
                        "value": "screen" if screen_idx == 1 else "real",
                        "confidence": round(screen_conf, 2),
                    }

            if payload.haze_classification:
                haze_pre = pp.preprocess_haze(img_rgb)
                haze_logits = self.models["haze"].infer(haze_pre)
                haze_idx = int(np.argmax(haze_logits))
                result["is_haze"] = haze_idx != 0

            if payload.check_cgi:
                cgi_pre = pp.preprocess_cgi(img_rgb)
                cgi_logits = self.models["cgi"].infer(cgi_pre)
                cgi_idx = int(np.argmax(cgi_logits))
                cgi_conf = float(self._softmax(cgi_logits)[cgi_idx]) * 100
                result["check_cgi"] = {
                    "value": "cgi" if cgi_idx == 0 else "non_cgi",
                    "confidence": round(cgi_conf, 2),
                }

            return CarValidationOutput(**result)

        except Exception:
            traceback.print_exc()
            # Return partial results on error — never crash the worker
            return CarValidationOutput(**result)