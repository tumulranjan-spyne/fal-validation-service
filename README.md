# Car Validation on fal.ai — Quick Start
# ==========================================

## 1. Prerequisites
- conda environment named `fal` with `fal` CLI (`pip install fal`)
- `fal auth login` (or set FAL_KEY env var)
- Your model files in a `validation_models/` directory (Triton layout)

## 2. Upload Models
```bash
export LOCAL_MODELS_DIR=/path/to/validation_models
./upload_models.sh
```

## 3. Deploy
```bash
conda activate fal
fal deploy car-validation
```

## 4. Test
```bash
curl -X POST https://fal.run/<workspace>/car-validation/validate_car \
  -H "Content-Type: application/json" \
  -H "Authorization: Key $FAL_KEY" \
  -d '{
    "image_url": "https://your-image.jpg",
    "car_cls": true,
    "car_type_cls": true,
    "angle_detect": true,
    "haze_classification": true
  }'
```

## 5. View Logs
```bash
fal apps logs car-validation
```

## Files
- `app.py` — fal.App with /validate_car endpoint
- `model_loader.py` — ONNX + TorchScript model loading
- `preprocessing.py` — per-model preprocessing
- `batcher.py` — dynamic request batching
- `Dockerfile` — container image
- `requirements.txt` — Python dependencies
- `pyproject.toml` — fal project config
- `upload_models.sh` — upload weights to fal storage