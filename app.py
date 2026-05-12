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
from pydantic import BaseModel, ConfigDict, model_validator

# Decode helpers live here (not a separate module) so fal isolate unpickle never hits
# ModuleNotFoundError for ``tensor_inputs`` when PYTHONPATH/cwd differs on workers.
import io as _io


def _ti_decode_npy_b64(b64: Optional[str]) -> Optional[Any]:
    if b64 is None or not str(b64).strip():
        return None
    np, _, _ = _np_cv_pp()
    raw = base64.standard_b64decode(str(b64).strip())
    return np.load(_io.BytesIO(raw), allow_pickle=False)


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

# Must match imagewizard CarAngleClassificationModel (auto-classification-angle_classification).
ANGLE_LABELS = [
    0, 10, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    20, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    30, 300, 310, 320, 330, 340, 350,
    40, 50, 60, 70, 80, 90,
]

GENERAL_LABELS = ['Human', 'Jewellery', 'Automobile', 'Footwear', 'Bag', 'Food', 'Food']
CAR_TYPE_LABELS = ["Hatchback", "SUV", "Sedan"]


# ============================================================================
# Request / Response schemas
# ============================================================================
class CarValidationInput(BaseModel):
    """POST body: flat JSON.

    **Image (required):** ``image_url`` *or* ``image_rgb_uint8_npy_b64`` (uint8 HWC RGB ``.npy``).

    **Optional per-model tensors** (base64 ``numpy.save`` each): skip matching preprocess,
    pass directly to ``infer()`` — shapes match ``preprocessing.py`` / Triton inputs:

    ``preprocessed_clip_vit_b32_224_uint8_npy_b64`` (224,224,3) uint8;
    ``precomputed_clip_embedding_npy_b64`` (512,) float — skips clip for ``car_cls``;
    ``preprocessed_roi_224_uint8_npy_b64``; ``preprocessed_davit_chw_fp32_npy_b64`` (3,512,512);
    ``preprocessed_segmentation_bgr768_uint8_npy_b64`` BGR 768³;
    ``precomputed_segmentation_mask_uint8_npy_b64`` (H,W) same as frame;
    ``precomputed_tp_rgba_uint8_npy_b64`` (H,W,4) skips segmentation;
    ``preprocessed_car_type_224_uint8_npy_b64``; ``preprocessed_angle_224_uint8_npy_b64``;
    ``preprocessed_reflection_224_uint8_npy_b64``; ``preprocessed_tyre_mud_224_uint8_npy_b64``;
    ``preprocessed_screen_224_uint8_npy_b64``; ``preprocessed_haze_224_uint8_npy_b64``;
    ``preprocessed_cgi_224_uint8_npy_b64`` — all (224,224,3) uint8 unless noted.
    """

    model_config = ConfigDict(extra="ignore")

    image_url: Optional[str] = None
    image_rgb_uint8_npy_b64: Optional[str] = None

    preprocessed_clip_vit_b32_224_uint8_npy_b64: Optional[str] = None
    precomputed_clip_embedding_npy_b64: Optional[str] = None
    preprocessed_roi_224_uint8_npy_b64: Optional[str] = None
    preprocessed_davit_chw_fp32_npy_b64: Optional[str] = None
    preprocessed_segmentation_bgr768_uint8_npy_b64: Optional[str] = None
    precomputed_segmentation_mask_uint8_npy_b64: Optional[str] = None
    precomputed_tp_rgba_uint8_npy_b64: Optional[str] = None
    preprocessed_car_type_224_uint8_npy_b64: Optional[str] = None
    preprocessed_angle_224_uint8_npy_b64: Optional[str] = None
    preprocessed_reflection_224_uint8_npy_b64: Optional[str] = None
    preprocessed_tyre_mud_224_uint8_npy_b64: Optional[str] = None
    preprocessed_screen_224_uint8_npy_b64: Optional[str] = None
    preprocessed_haze_224_uint8_npy_b64: Optional[str] = None
    preprocessed_cgi_224_uint8_npy_b64: Optional[str] = None

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

    @model_validator(mode="after")
    def _exactly_one_image_source(self) -> "CarValidationInput":
        has_url = bool(self.image_url and str(self.image_url).strip())
        has_arr = bool(
            self.image_rgb_uint8_npy_b64 and str(self.image_rgb_uint8_npy_b64).strip()
        )
        if has_url and has_arr:
            raise ValueError(
                "Provide either image_url or image_rgb_uint8_npy_b64, not both"
            )
        if not has_url and not has_arr:
            raise ValueError(
                "Provide image_url or image_rgb_uint8_npy_b64 (uint8 HWC RGB as npy base64)"
            )
        return self


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

    # Ordered fallback if 5090 capacity is tight (see fal-validation/deploy/USAGE.txt).
    machine_type = ["GPU-RTX5090", "GPU-A6000", "GPU-A100"]
    image = ContainerImage.from_dockerfile("Dockerfile")
    # Single runner: min=max=1. Avoid repeated setup() by keeping this process hot long after
    # the last request (see https://fal.ai/documentation/deployment/scale-your-application).
    # New setup() only runs when fal starts a *new* runner (deploy, crash, host maintenance).
    keep_alive = 3600  # seconds idle retention before scaling to zero
    min_concurrency = 1
    max_concurrency = 1
    scaling_delay = 0  # was 120: with capacity at 0, do not wait 2min before starting a runner
    # Many ONNX sessions + TorchScript on cold GPU; default 600s often kills mid-startup.
    startup_timeout = 3600
    request_timeout = 3600
    # Default max_multiplexing=1 ⇒ one request at a time on this GPU. Throughput for 10k+
    # requests is ~ 10000 * latency; increase only if handlers are async-safe and VRAM allows.

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
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._ensure_min_rgb_size(rgb, context=url[:120])

    @staticmethod
    def _decode_rgb_uint8_npy_b64(b64: str) -> Any:
        import io

        np, _, _ = _np_cv_pp()
        raw = base64.standard_b64decode(b64.strip())
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"image_rgb_uint8_npy_b64 must decode to uint8 HWC RGB; got {arr.shape} {arr.dtype}"
            )
        return np.ascontiguousarray(arr)

    @staticmethod
    def _ensure_min_rgb_size(rgb: Any, *, context: str = "") -> Any:
        """Match post-decode safeguards for URL path; ``rgb`` is HxWx3 uint8."""
        np, cv2, _ = _np_cv_pp()
        h, w = rgb.shape[:2]
        if h < 1 or w < 1:
            raise ValueError(f"Degenerate image size {w}x{h} {context}".strip())
        min_side = min(h, w)
        if min_side < 32:
            scale = 32.0 / float(min_side)
            rgb = cv2.resize(
                rgb,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        return rgb

    def _rgb_from_payload(self, payload: CarValidationInput) -> Any:
        if payload.image_rgb_uint8_npy_b64:
            rgb = self._decode_rgb_uint8_npy_b64(payload.image_rgb_uint8_npy_b64)
            return self._ensure_min_rgb_size(rgb, context="(npy)")
        return self._download_rgb(payload.image_url)  # type: ignore[arg-type]

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
    def _classify_roi(self, img_rgb: Any, payload: CarValidationInput) -> Tuple[str, Optional[str], float]:
        """Returns (roi_label, subroi_label, confidence)."""
        np, _, pp = _np_cv_pp()
        t_roi = _ti_decode_npy_b64(payload.preprocessed_roi_224_uint8_npy_b64)
        if t_roi is not None:
            pre_roi = np.ascontiguousarray(t_roi.astype(np.uint8))
            if pre_roi.shape != (224, 224, 3):
                raise ValueError(f"preprocessed_roi must be (224,224,3) uint8; got {pre_roi.shape}")
        else:
            pre_roi = pp.preprocess_roi(img_rgb)
        logits = np.squeeze(self.models["roi"].infer(pre_roi)).astype(np.float32)
        idx = int(np.argmax(logits))
        coarse = COARSE_ROI_LABELS[idx]
        conf = float(self._softmax(logits)[idx])

        if coarse == "outer":
            return coarse, None, conf

        t_dav = _ti_decode_npy_b64(payload.preprocessed_davit_chw_fp32_npy_b64)
        if t_dav is not None:
            davit_in = np.ascontiguousarray(t_dav.astype(np.float32))
            if davit_in.ndim == 4 and davit_in.shape[0] == 1:
                davit_in = davit_in[0]
            if davit_in.shape != (3, 512, 512):
                raise ValueError(
                    f"preprocessed_davit must be (3,512,512) fp32; got {davit_in.shape}"
                )
        else:
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
    def _general_classify(self, img_rgb: Any, payload: CarValidationInput) -> Tuple[str, float]:
        np, _, pp = _np_cv_pp()
        t_emb = _ti_decode_npy_b64(payload.precomputed_clip_embedding_npy_b64)
        if t_emb is not None:
            emb = np.squeeze(t_emb).astype(np.float32)
            if emb.size != 512:
                emb = emb.reshape(-1)
            if emb.shape[0] != 512:
                raise ValueError(
                    f"precomputed_clip_embedding must flatten to 512; got {t_emb.shape}"
                )
            emb_in = emb.astype(np.float32)
        else:
            t_clip = _ti_decode_npy_b64(payload.preprocessed_clip_vit_b32_224_uint8_npy_b64)
            if t_clip is not None:
                pre = np.ascontiguousarray(t_clip.astype(np.uint8))
                if pre.shape != (224, 224, 3):
                    raise ValueError(f"preprocessed_clip must be (224,224,3); got {pre.shape}")
            else:
                pre = pp.preprocess_clip(img_rgb, center_crop=True)
            emb_out = self.models["clip_backbone"].infer(pre)
            emb_in = np.squeeze(emb_out).astype(np.float32)
        logits = np.squeeze(self.models["general_classifier"].infer(emb_in)).astype(np.float64)
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
            img_rgb = self._rgb_from_payload(payload)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # ---- Step 1: General classification ----
            if payload.car_cls:
                cat, conf = self._general_classify(img_rgb, payload)
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
                raw_roi, _, roi_conf = self._classify_roi(img_rgb, payload)

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
                    _, sub_label, sub_conf = self._classify_roi(img_rgb, payload)
                    result["interior_class"] = {
                        "value": sub_label,
                        "confidence": round(abs(sub_conf), 2),
                    }
                    result["sub_category"] = dict(result["interior_class"])
                elif roi_display == "Focus":
                    _, sub_label, sub_conf = self._classify_roi(img_rgb, payload)
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
            angle_val: Optional[int] = None
            if roi_display == "Exterior":
                result["number_plate"] = None

                # Segmentation / tp_rgba (optional client tensors skip compute)
                t_tp = _ti_decode_npy_b64(payload.precomputed_tp_rgba_uint8_npy_b64)
                t_msk = _ti_decode_npy_b64(payload.precomputed_segmentation_mask_uint8_npy_b64)
                t_seg768 = _ti_decode_npy_b64(payload.preprocessed_segmentation_bgr768_uint8_npy_b64)

                if t_tp is not None:
                    tp_rgba = np.ascontiguousarray(t_tp.astype(np.uint8))
                    if tp_rgba.ndim != 4 or tp_rgba.shape[2] != 4:
                        raise ValueError(f"precomputed_tp_rgba must be HxWx4 uint8; got {tp_rgba.shape}")
                    if tp_rgba.shape[0] != h_img or tp_rgba.shape[1] != w_img:
                        raise ValueError(
                            f"tp_rgba H×W {tp_rgba.shape[:2]} must match image {h_img}×{w_img}"
                        )
                    mask = tp_rgba[:, :, 3]
                elif t_msk is not None:
                    mask = np.ascontiguousarray(t_msk.astype(np.uint8))
                    if mask.ndim != 2 or mask.shape != (h_img, w_img):
                        raise ValueError(
                            f"precomputed_segmentation_mask must be (H,W)=({h_img},{w_img}); got {mask.shape}"
                        )
                    tp_rgba = np.dstack([img_rgb, mask])
                else:
                    if t_seg768 is not None:
                        b768 = np.ascontiguousarray(t_seg768.astype(np.uint8))
                        if b768.shape != (768, 768, 3):
                            raise ValueError(
                                f"preprocessed_segmentation_bgr768 must be (768,768,3); got {b768.shape}"
                            )
                        seg_pixel = b768
                    else:
                        seg_pixel = pp.preprocess_segmentation(img_bgr)[0]
                    mask_arr = self.models["segmentation"].infer(seg_pixel)
                    mask = mask_arr.squeeze().astype(np.uint8)
                    if mask.shape != img_bgr.shape[:2]:
                        mask = cv2.resize(
                            mask,
                            (img_bgr.shape[1], img_bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    mask = (mask > 128).astype(np.uint8) * 255
                    tp_rgba = np.dstack([img_rgb, mask])

                # Car type
                if payload.car_type_cls:
                    t_ct = _ti_decode_npy_b64(payload.preprocessed_car_type_224_uint8_npy_b64)
                    if t_ct is not None:
                        type_pre = np.ascontiguousarray(t_ct.astype(np.uint8))
                        if type_pre.shape != (224, 224, 3):
                            raise ValueError(f"preprocessed_car_type must be (224,224,3); got {type_pre.shape}")
                    else:
                        type_pre = pp.preprocess_car_type(img_rgb, mask)
                    type_logits = self.models["car_type"].infer(type_pre)
                    type_logits = np.squeeze(type_logits).astype(np.float64)
                    if type_logits.ndim != 1:
                        type_logits = type_logits.reshape(-1)
                    type_idx = int(np.argmax(type_logits))
                    if type_idx < 0 or type_idx >= len(CAR_TYPE_LABELS):
                        raise ValueError(
                            f"car_type model index {type_idx} out of range (n={len(type_logits)})"
                        )
                    type_conf = float(self._softmax(type_logits)[type_idx]) * 100
                    result["car_type"] = {
                        "value": CAR_TYPE_LABELS[type_idx],
                        "confidence": round(type_conf, 2),
                    }

                # Angle
                if payload.angle_detect:
                    t_ang = _ti_decode_npy_b64(payload.preprocessed_angle_224_uint8_npy_b64)
                    if t_ang is not None:
                        angle_pre = np.ascontiguousarray(t_ang.astype(np.uint8))
                        if angle_pre.shape != (224, 224, 3):
                            raise ValueError(f"preprocessed_angle must be (224,224,3); got {angle_pre.shape}")
                    else:
                        angle_pre = pp.preprocess_angle(tp_rgba)
                    angle_logits = self.models["angle"].infer(angle_pre)
                    angle_logits = np.squeeze(angle_logits).astype(np.float64)
                    if angle_logits.ndim != 1:
                        angle_logits = angle_logits.reshape(-1)
                    angle_idx = int(np.argmax(angle_logits))
                    if angle_idx < 0 or angle_idx >= len(ANGLE_LABELS):
                        raise ValueError(
                            f"angle model index {angle_idx} out of range (n={len(angle_logits)})"
                        )
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
                    t_rf = _ti_decode_npy_b64(payload.preprocessed_reflection_224_uint8_npy_b64)
                    if t_rf is not None:
                        refl_pre = np.ascontiguousarray(t_rf.astype(np.uint8))
                        if refl_pre.shape != (224, 224, 3):
                            raise ValueError(f"preprocessed_reflection must be (224,224,3); got {refl_pre.shape}")
                    else:
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
                    t_md = _ti_decode_npy_b64(payload.preprocessed_tyre_mud_224_uint8_npy_b64)
                    if t_md is not None:
                        mud_pre = np.ascontiguousarray(t_md.astype(np.uint8))
                        if mud_pre.shape != (224, 224, 3):
                            raise ValueError(f"preprocessed_tyre_mud must be (224,224,3); got {mud_pre.shape}")
                    else:
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
                    t_sc = _ti_decode_npy_b64(payload.preprocessed_screen_224_uint8_npy_b64)
                    if t_sc is not None:
                        screen_pre = np.ascontiguousarray(t_sc.astype(np.uint8))
                        if screen_pre.shape != (224, 224, 3):
                            raise ValueError(f"preprocessed_screen must be (224,224,3); got {screen_pre.shape}")
                    else:
                        screen_pre = pp.preprocess_224_pil(img_rgb)
                    screen_logits = self.models["screen_classifier"].infer(screen_pre)
                    screen_idx = int(np.argmax(screen_logits))
                    screen_conf = float(self._softmax(screen_logits)[screen_idx]) * 100
                    result["screen_detect"] = {
                        "value": "screen" if screen_idx == 1 else "real",
                        "confidence": round(screen_conf, 2),
                    }

            if payload.haze_classification:
                t_hz = _ti_decode_npy_b64(payload.preprocessed_haze_224_uint8_npy_b64)
                if t_hz is not None:
                    haze_pre = np.ascontiguousarray(t_hz.astype(np.uint8))
                    if haze_pre.shape != (224, 224, 3):
                        raise ValueError(f"preprocessed_haze must be (224,224,3); got {haze_pre.shape}")
                else:
                    haze_pre = pp.preprocess_haze(img_rgb)
                haze_logits = self.models["haze"].infer(haze_pre)
                haze_idx = int(np.argmax(haze_logits))
                result["is_haze"] = haze_idx != 0

            if payload.check_cgi:
                t_cg = _ti_decode_npy_b64(payload.preprocessed_cgi_224_uint8_npy_b64)
                if t_cg is not None:
                    cgi_pre = np.ascontiguousarray(t_cg.astype(np.uint8))
                    if cgi_pre.shape != (224, 224, 3):
                        raise ValueError(f"preprocessed_cgi must be (224,224,3); got {cgi_pre.shape}")
                else:
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