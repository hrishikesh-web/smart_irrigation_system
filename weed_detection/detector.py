"""
Weed / crop detector used by the Streamlit dashboard (Stage 3).

- Loads ONLY the model trained by weed_detection/train.py:  weed_detection/models/weed_detector.pt
- Has no Streamlit code, so it can be tested on its own (see smoke_test_detector.py).
- Imports Ultralytics only when a model is actually loaded, so importing this file
  (or dashboard.py) stays fast and never fails just because Ultralytics is missing.

What the model can do: draw boxes for two classes, "crop" and "weed", as defined by the
training dataset (one crop species photographed on bare soil). It does NOT identify weed
species, diagnose disease, judge crop health or measure growth.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

try:  # imported as part of the package (dashboard:  from weed_detection import detector)
    from . import weed_settings as cfg
except ImportError:  # run directly as a script from inside weed_detection/
    import weed_settings as cfg

ALLOWED_FORMATS = {"JPEG", "PNG"}         # what Pillow reports for .jpg/.jpeg and .png files
MAX_PIXELS = 40_000_000                   # refuse absurdly large images (about 7000 x 5700)
MAX_BYTES = 25 * 1024 * 1024              # 25 MB upload limit


class WeedModelError(RuntimeError):
    """The model file is missing, unreadable, or not our crop/weed model."""


class WeedImageError(ValueError):
    """The uploaded file is not a usable JPG/PNG image."""


# ----------------------------------------------------------------------------- model

def model_status(path: Path = cfg.FINAL_MODEL) -> dict:
    """Cheap check (no loading) used by the dashboard to show a friendly status line."""
    path = Path(path)
    return {"path": str(path), "exists": path.is_file(),
            "size_mb": round(path.stat().st_size / 1e6, 1) if path.is_file() else None,
            "modified": path.stat().st_mtime if path.is_file() else None}


def load_model(path: Path = cfg.FINAL_MODEL):
    """Load the trained YOLO model and confirm it is the crop/weed model."""
    path = Path(path)
    if not path.is_file():
        raise WeedModelError(
            f"Trained model not found at {path}. Train it first with:  python weed_detection\\train.py")
    if cfg.WEED_DIR.resolve() not in path.resolve().parents or path.parent.resolve() == cfg.PRETRAINED_DIR.resolve():
        raise WeedModelError(f"Refusing to load {path}: only models trained by weed_detection/train.py are allowed.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise WeedModelError("Ultralytics is not installed in this environment "
                             "(python -m pip install ultralytics).") from exc
    try:
        model = YOLO(str(path))
    except Exception as exc:  # noqa: BLE001 - corrupted / incompatible file
        raise WeedModelError(f"Could not load the model file {path.name}: {exc}") from exc
    names = {int(k): v for k, v in model.names.items()}
    if names != cfg.EXPECTED_NAMES:
        raise WeedModelError(f"{path.name} has classes {names}, expected {cfg.EXPECTED_NAMES}.")
    return model


# ----------------------------------------------------------------------------- image

def read_image(data: bytes) -> Image.Image:
    """Turn uploaded bytes into an upright RGB image, or raise WeedImageError."""
    if not data:
        raise WeedImageError("The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise WeedImageError(f"The image is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            probe.verify()                       # catches truncated / corrupted files
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise WeedImageError("This file is not a readable JPG or PNG image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise WeedImageError(f"Unsupported image type '{fmt}'. Please upload a JPG, JPEG or PNG.")
    if w * h > MAX_PIXELS:
        raise WeedImageError(f"Image is too large ({w} x {h} pixels).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)       # phone photos: apply the rotation stored in EXIF
        return img.convert("RGB")                # drops PNG transparency / greyscale
    except Exception as exc:  # noqa: BLE001
        raise WeedImageError(f"Could not decode the image: {exc}") from exc


# ----------------------------------------------------------------------------- detection

def detect(model, data: bytes, conf: float = cfg.DEFAULT_CONFIDENCE,
           imgsz: int = cfg.TRAIN_DEFAULTS["imgsz"]) -> dict:
    """
    Run the model on one uploaded image.

    Returns a dictionary:
        annotated      PIL image with boxes, class labels and confidence scores
        detections     list of {"class", "confidence", "box_xyxy"} sorted by confidence
        weed_count, crop_count, max_weed_confidence (0-1 or None), image_size, confidence_threshold
    Raises WeedImageError for bad images and RuntimeError if inference itself fails.
    """
    if not 0.0 < conf < 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    img = read_image(data)
    try:
        result = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0]
        annotated = result.plot(pil=True, conf=True, labels=True)
    except Exception as exc:  # noqa: BLE001 - e.g. CUDA out of memory, driver problems
        raise RuntimeError(f"Detection failed: {exc}") from exc

    names = {int(k): v for k, v in model.names.items()}
    detections = []
    if result.boxes is not None and len(result.boxes):
        for cls, score, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                    result.boxes.xyxy.tolist()):
            detections.append({"class": names[int(cls)], "confidence": round(float(score), 4),
                               "box_xyxy": [round(float(v), 1) for v in xyxy]})
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    weeds = [d for d in detections if d["class"] == "weed"]
    return {
        "annotated": annotated if isinstance(annotated, Image.Image) else Image.fromarray(annotated),
        "detections": detections,
        "weed_count": len(weeds),
        "crop_count": sum(d["class"] == "crop" for d in detections),
        "max_weed_confidence": weeds[0]["confidence"] if weeds else None,
        "image_size": img.size,
        "confidence_threshold": conf,
    }

