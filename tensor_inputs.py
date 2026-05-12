"""Optional per-model tensor payloads (base64 npy) for validate_car — see CarValidationInput docstring."""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np


def decode_npy_b64(b64: Optional[str]) -> Optional[np.ndarray]:
    if b64 is None or not str(b64).strip():
        return None
    raw = base64.standard_b64decode(str(b64).strip())
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")
