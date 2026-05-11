"""
Preprocessing functions for car validation models.
Mirrors imagewizard preprocessing exactly.
"""
import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CLIP ViT-B/32 backbone
# ---------------------------------------------------------------------------
def preprocess_clip(image: np.ndarray, center_crop: bool = True) -> np.ndarray:
    """RGB uint8 -> [224,224,3] uint8 with aspect-preserving center crop."""
    if center_crop:
        h, w = image.shape[:2]
        scale = 224 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        return resized[start_h:start_h + 224, start_w:start_w + 224]
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# auto-classification-roi
# ---------------------------------------------------------------------------
def preprocess_roi(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> [224,224,3] uint8 via PIL LANCZOS."""
    pil = Image.fromarray(image).resize((224, 224), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# auto-classification-roi-davit
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_davit(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> FP32 [1,3,512,512] NCHW, ImageNet normalized."""
    pil = Image.fromarray(image).resize((512, 512), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# auto-classification-angle_classification
# ---------------------------------------------------------------------------
def preprocess_angle(tp_rgba: np.ndarray) -> np.ndarray:
    """
    RGBA (car on black bg) -> [224,224,3] uint8 grayscale triple-channel.
    Matches CarAngleClassificationModel.process_image.
    """
    alpha = tp_rgba[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    pad = int(0.2 * max(x_max - x_min, y_max - y_min))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(tp_rgba.shape[1], x_max + pad)
    y_max = min(tp_rgba.shape[0], y_max + pad)

    crop = tp_rgba[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    return cv2.resize(gray3, (224, 224), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# car_type_classifier_clip
# ---------------------------------------------------------------------------
def preprocess_car_type(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    RGB + mask -> [224,224,3] uint8 grayscale triple-channel.
    Matches CarTypeClassificationModel.__call__.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray[mask < 20] = 0
    ys, xs = np.nonzero(gray > 0)
    if len(xs) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    crop = gray[y_min:y_max, x_min:x_max]
    crop = cv2.copyMakeBorder(crop, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return cv2.merge([resized, resized, resized])


# ---------------------------------------------------------------------------
# auto-segmentation-exterior_removebg
# ---------------------------------------------------------------------------
def preprocess_segmentation(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> [1,768,768,3] uint8 bilinear resize."""
    resized = cv2.resize(bgr, (768, 768), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized, axis=0)


# ---------------------------------------------------------------------------
# Generic 224x224 PIL resize (haze, cgi, focus, resnet)
# ---------------------------------------------------------------------------
def preprocess_224_pil(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> [224,224,3] uint8 via PIL LANCZOS."""
    pil = Image.fromarray(image).resize((224, 224), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


# Aliases
preprocess_haze   = preprocess_224_pil
preprocess_cgi    = preprocess_224_pil
preprocess_focus  = preprocess_224_pil
preprocess_resnet = preprocess_224_pil


# ---------------------------------------------------------------------------
# Reflection (crop from tp_image, matching ReflectionClassifier)
# ---------------------------------------------------------------------------
def preprocess_reflection_tp(tp_rgba: np.ndarray) -> np.ndarray:
    """RGBA -> [224,224,3] uint8, zeroing RGB where alpha < 20, bbox crop."""
    img = tp_rgba[:, :, :3].copy()
    mask = tp_rgba[:, :, 3]
    img[mask < 20] = 0
    ys, xs = np.nonzero(np.any(img > 0, axis=2))
    if len(xs) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    crop = img[y1:y2 + 1, x1:x2 + 1]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)