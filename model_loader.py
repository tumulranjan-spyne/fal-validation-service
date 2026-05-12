"""
Model loader — loads all ONNX and TorchScript models from /data.

Models are loaded lazily on first use to avoid startup timeouts.
Each model is wrapped in a DynamicBatcher for throughput.
"""
import os
import time
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Any, Dict, Optional, Callable
from batcher import DynamicBatcher

DATA_DIR = os.environ.get("CAR_VALIDATION_DATA_DIR", "/data/validation_models")


def list_required_weight_paths(root: Optional[str] = None) -> list[str]:
    """Paths that must exist (Triton-style layout); excludes optional screen_classification."""
    base = root if root is not None else DATA_DIR
    parts = (
        ("auto-classification-roi", "1", "model.onnx"),
        ("auto-classification-roi-davit", "1", "model.onnx"),
        ("auto-classification-angle_classification", "1", "model.onnx"),
        ("cgi_classifier", "1", "model.onnx"),
        ("haze_classifier", "1", "model.onnx"),
        ("clip_vit_B32", "1", "model.pt"),
        ("general_classifier", "1", "model.pt"),
        ("car_type_classifier_clip", "1", "model.pt"),
        ("auto-segmentation-exterior_removebg", "1", "model.pt"),
        ("focus_classifier", "1", "model.pt"),
        ("resnet_backbone_clf", "1", "model.pt"),
        ("resnet_backbone_clf", "2", "model.pt"),
    )
    return [os.path.join(base, *p) for p in parts]


def wait_for_validation_weights(weight_dir: str) -> None:
    """
    After bulk fal uploads, ``/data`` can be briefly empty on cold runners — optional poll:

      CAR_VALIDATION_WEIGHT_SYNC_WAIT_SEC (seconds, default 0)
      CAR_VALIDATION_WEIGHT_SYNC_INTERVAL_SEC (seconds, default 5)

    Only runs when ``weight_dir`` is under fal persistent ``/data/...``.
    """
    sync_wait = int(os.environ.get("CAR_VALIDATION_WEIGHT_SYNC_WAIT_SEC", "0").strip() or "0")
    interval = max(
        1,
        int(os.environ.get("CAR_VALIDATION_WEIGHT_SYNC_INTERVAL_SEC", "5").strip() or "5"),
    )
    wd = os.path.abspath(weight_dir)
    if sync_wait <= 0 or not (wd == "/data" or wd.startswith("/data/")):
        return

    deadline = time.monotonic() + sync_wait

    def missing() -> list[str]:
        return [p for p in list_required_weight_paths(weight_dir) if not os.path.isfile(p)]

    pending = missing()
    while pending and time.monotonic() < deadline:
        samples = sorted({os.path.basename(p) for p in pending[:8]})
        print(
            f"[model] weights not ready ({len(pending)} missing, e.g. {samples}); "
            f"sync wait up to {sync_wait}s, interval={interval}s"
        )
        try:
            entries = os.listdir(weight_dir)
            print(f"[model] listdir({weight_dir!r}) count={len(entries)} sample={entries[:25]!r}")
        except OSError as e:
            print(f"[model] listdir failed: {e}")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        pending = missing()
    if missing():
        raise FileNotFoundError(
            f"Weights still missing under {weight_dir!r}; "
            "run upload_models.sh and/or increase CAR_VALIDATION_WEIGHT_SYNC_WAIT_SEC"
        )


# ============================================================================
# ONNX Model (CUDA EP, no TensorRT to avoid engine compilation timeouts)
# ============================================================================
class ONNXModel:
    def __init__(self, model_path: str, input_name: str, output_name: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        force_cpu = os.environ.get("CAR_VALIDATION_ORT_CPU", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # ERROR only

        # In fal/docker with cgroup cpu limits ORT tries invalid pthread_setaffinity masks;
        # setting thread counts skips that path (matches ORT log hint).
        intra = max(1, int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "2").strip() or "2"))
        inter = max(1, int(os.environ.get("ORT_INTER_OP_NUM_THREADS", "1").strip() or "1"))
        opts.intra_op_num_threads = intra
        opts.inter_op_num_threads = inter

        provider_attempts = (
            [["CPUExecutionProvider"]]
            if force_cpu
            else [["CUDAExecutionProvider"], ["CPUExecutionProvider"]]
        )

        last_err: Optional[Exception] = None
        self.session = None  # type: ignore[assignment]
        for providers in provider_attempts:
            try:
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
                break
            except Exception as e:
                last_err = e
                continue

        if self.session is None:
            raise RuntimeError(
                f"Failed to create ONNX session for {model_path}"
            ) from last_err

        self.input_name = input_name
        self.output_name = output_name
        print(f"[model] ONNX {os.path.basename(model_path)} "
              f"providers={self.session.get_providers()}")

    def __call__(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        stacked = np.stack(batch, axis=0)
        out = self.session.run([self.output_name], {self.input_name: stacked})[0]
        return [out[i] for i in range(out.shape[0])]


# ============================================================================
# TorchScript Model
# ============================================================================
class TorchScriptModel:
    def __init__(
        self,
        model_path: str,
        input_dtype: Optional[torch.dtype] = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        # Traced CLIP-style heads often bake matmul weights in float16 while activations
        # default to float32 from numpy — matmul then raises float vs Half.
        self.input_dtype = input_dtype
        print(f"[model] TorchScript {os.path.basename(model_path)} "
              f"device={self.device} input_dtype={input_dtype!r}")

    def __call__(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        stacked = np.stack(batch, axis=0)
        tensor = torch.from_numpy(stacked).to(self.device)
        if self.input_dtype is not None:
            tensor = tensor.to(self.input_dtype)
        with torch.no_grad():
            out = self.model(tensor)
        out_np = out.cpu().numpy()
        return [out_np[i] for i in range(out_np.shape[0])]


# ============================================================================
# Model Registry
# ============================================================================
def load_all_models() -> Dict[str, DynamicBatcher]:
    """Load all models. Returns dict of model_name -> DynamicBatcher."""
    wait_for_validation_weights(DATA_DIR)
    t0 = time.perf_counter()
    models: Dict[str, DynamicBatcher] = {}

    def add(key: str, factory: Callable[[], Any], max_batch: int, delay_ms: int):
        print(f"[model] loading {key} ...")
        backend = factory()
        models[key] = DynamicBatcher(max_batch, delay_ms, backend)

    # --- ONNX models ---
    add("roi", lambda: ONNXModel(
        os.path.join(DATA_DIR, "auto-classification-roi", "1", "model.onnx"),
        "input", "output"), max_batch=64, delay_ms=5)

    add("davit", lambda: ONNXModel(
        os.path.join(DATA_DIR, "auto-classification-roi-davit", "1", "model.onnx"),
        "INPUT__0", "OUTPUT__0"), max_batch=32, delay_ms=10)

    add("angle", lambda: ONNXModel(
        os.path.join(DATA_DIR, "auto-classification-angle_classification", "1", "model.onnx"),
        "input", "output"), max_batch=64, delay_ms=5)

    add("cgi", lambda: ONNXModel(
        os.path.join(DATA_DIR, "cgi_classifier", "1", "model.onnx"),
        "input", "output"), max_batch=16, delay_ms=10)

    add("haze", lambda: ONNXModel(
        os.path.join(DATA_DIR, "haze_classifier", "1", "model.onnx"),
        "input", "output"), max_batch=16, delay_ms=10)

    # --- TorchScript models ---
    add("clip_backbone", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "clip_vit_B32", "1", "model.pt")),
        max_batch=64, delay_ms=5)

    add("general_classifier", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "general_classifier", "1", "model.pt"),
        input_dtype=torch.float16,
    ),
        max_batch=64, delay_ms=5)

    add("car_type", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "car_type_classifier_clip", "1", "model.pt"),
        input_dtype=torch.float16,
    ),
        max_batch=64, delay_ms=5)

    add("segmentation", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "auto-segmentation-exterior_removebg", "1", "model.pt")),
        max_batch=8, delay_ms=20)

    add("focus", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "focus_classifier", "1", "model.pt")),
        max_batch=8, delay_ms=10)

    add("resnet_v1", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "resnet_backbone_clf", "1", "model.pt")),
        max_batch=16, delay_ms=10)

    add("resnet_v2", lambda: TorchScriptModel(
        os.path.join(DATA_DIR, "resnet_backbone_clf", "2", "model.pt")),
        max_batch=16, delay_ms=10)

    # Optional: screen_classifier
    screen_path = os.path.join(DATA_DIR, "screen_classification", "1", "model.onnx")
    if os.path.isfile(screen_path):
        add("screen_classifier", lambda: ONNXModel(
            screen_path, "input", "output"), max_batch=16, delay_ms=10)
    else:
        print("[model] screen_classification/1/model.onnx not found — screen_detect disabled")

    print(f"[model] {len(models)} models loaded in {time.perf_counter() - t0:.1f}s")
    return models