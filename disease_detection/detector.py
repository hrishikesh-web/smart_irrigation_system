"""
Smart Agriculture — AI Crop Disease Detection Engine
=====================================================
Modular inference engine for tomato leaf disease detection using YOLOv8 classification.
Supports model loading, confidence thresholding, top-k alternative predictions,
uncertainty handling, and informational agronomic advisories.
"""

import io
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "training" / "disease_detection" / "weights" / "best.pt"
METADATA_PATH = PROJECT_ROOT / "training" / "disease_detection" / "model_metadata.json"

DEFAULT_CONF_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.15

# Crop-specific informational agronomic advisories (non-prescriptive)
ADVISORY_GUIDANCE = {
    "Healthy": {
        "status": "Optimal",
        "description": "Foliage displays uniform green coloration and intact structural morphology without observable necrotic or chlorotic lesions.",
        "cultural_measures": [
            "Maintain current balanced fertigation and moisture management schedule.",
            "Continue periodic scouting of lower canopy leaves for early symptom emergence.",
            "Maintain adequate spacing between rows to promote light penetration and airflow.",
        ],
        "monitoring": "Routine scouting every 3–5 days during peak vegetative and fruiting stages.",
    },
    "Early Blight": {
        "status": "Pathological Warning",
        "pathogen": "Alternaria solani (Fungus)",
        "description": "Characterized by dark brown to black necrotic lesions with distinctive concentric rings ('target board' pattern), typically initiating on older, lower foliage.",
        "cultural_measures": [
            "Prune and safely destroy lower leaves touching the soil line.",
            "Avoid overhead irrigation; utilize drip lines to minimize leaf wetness duration.",
            "Apply organic mulch around plant bases to prevent spore splash from infested soil.",
            "Ensure wide plant spacing and staking to maximize canopy ventilation.",
            "Implement a 2–3 year crop rotation excluding solanaceous crops (tomatoes, potatoes, eggplants).",
        ],
        "monitoring": "Inspect lower leaves daily during warm (24–29°C), humid weather conditions.",
    },
    "Late Blight": {
        "status": "High Alert",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "description": "Highly aggressive water mold inducing irregular, dark, water-soaked lesions that rapidly expand and turn brown/purplish, often accompanied by white mycelial sporulation on leaf undersides in humid air.",
        "cultural_measures": [
            "Immediately rogued and bag heavily infected plants; do not compost infected material.",
            "Eliminate overhead watering completely; water only at root zones early in the morning.",
            "Reduce canopy humidity through aggressive ventilation and wider row spacing.",
            "Sanitize trellising wires, pruning shears, and footwear between plot rows.",
            "Consult local agricultural extension specialists promptly if regional late blight alerts are active.",
        ],
        "monitoring": "Urgent monitoring required every 24–48 hours under cool, overcast, damp conditions (15–22°C, >90% RH).",
    },
    "Septoria Leaf Spot": {
        "status": "Pathological Warning",
        "pathogen": "Septoria lycopersici (Fungus)",
        "description": "Presents as numerous small circular spots (2–5 mm) with dark brown margins and gray/tan centers, frequently exhibiting tiny black specks (pycnidia fruiting bodies) inside the lesions.",
        "cultural_measures": [
            "Strip infected lower leaves as soon as first spots appear to slow upward disease progression.",
            "Apply straw or plastic mulch to create a physical barrier preventing rain/splash dispersal.",
            "Sterilize shears in 70% isopropyl alcohol between consecutive pruning cuts.",
            "Refrain from handling or harvesting tomato plants while foliage is wet from dew or rain.",
            "Destroy all solanaceous crop residues thoroughly at the end of the harvest cycle.",
        ],
        "monitoring": "Scout foliage twice weekly following warm, wet spells.",
    },
}

SAFETY_DISCLAIMER = (
    "Agricultural Advisory Notice: This computer vision classification is an experimental "
    "decision-support aid trained on single-leaf specimens against controlled neutral backgrounds. "
    "It does NOT provide an automatic chemical prescription or fungicide recommendation. "
    "Always confirm suspected symptoms via in-field physical inspection and consult a certified "
    "local agronomist or agricultural extension officer before applying any crop protection products."
)


class DiseaseImageError(Exception):
    """Raised when an uploaded crop image cannot be decoded, opened, or processed."""
    pass


def model_status() -> Dict[str, Any]:
    """
    Check availability, size, classes, and metadata of the disease detection checkpoint.
    """
    exists = MODEL_PATH.exists()
    size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if exists else 0.0
    mod_time = MODEL_PATH.stat().st_mtime if exists else 0.0

    metadata = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    classes = metadata.get("classes", ["Early Blight", "Healthy", "Late Blight", "Septoria Leaf Spot"])

    return {
        "exists": exists,
        "path": MODEL_PATH,
        "size_mb": size_mb,
        "modified": mod_time,
        "classes": classes,
        "num_classes": len(classes),
        "architecture": metadata.get("model_architecture", "YOLOv8n-cls"),
        "val_accuracy": metadata.get("val_metrics", {}).get("accuracy_top1"),
        "device": metadata.get("device_used", "mps"),
        "version": "1.0.0-experimental",
    }


def load_model(path: Optional[Union[str, Path]] = None) -> YOLO:
    """
    Load YOLOv8 classification model from disk.
    """
    target_path = Path(path) if path else MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained crop disease model not found at '{target_path}'. "
            "Please run: python training/train_disease_detector.py"
        )
    return YOLO(str(target_path))


def predict_disease(
    model: YOLO,
    image_input: Union[bytes, Image.Image, Path, str],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run disease classification on input image.

    Args:
        model: Loaded YOLO classification model.
        image_input: Raw image bytes, PIL Image, or file path.
        conf_threshold: Minimum confidence required before flagging as uncertain.
        margin_threshold: Minimum gap between top-1 and top-2 predictions.

    Returns:
        Structured prediction dictionary with predicted class, confidence,
        top alternatives, uncertainty evaluation, and agronomic advisory.
    """
    # 1. Parse and decode image
    pil_image = None
    try:
        if isinstance(image_input, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        w, h = pil_image.size
        if w < 16 or h < 16:
            raise DiseaseImageError(f"Image dimensions ({w}x{h}) are too small for valid leaf analysis.")
    except Exception as exc:
        if isinstance(exc, DiseaseImageError):
            raise
        raise DiseaseImageError(f"Unable to decode image: {exc}")

    # 2. Run inference
    try:
        results = model(pil_image, verbose=False)
        r = results[0]
        probs = r.probs.data.cpu().numpy()
    except Exception as exc:
        raise RuntimeError(f"YOLO disease classification inference failed: {exc}")

    class_map = model.names
    sorted_indices = np.argsort(probs)[::-1]

    top_idx = int(sorted_indices[0])
    top_class = class_map[top_idx]
    top_conf = float(probs[top_idx])

    # Top predictions list
    top_predictions = []
    for idx in sorted_indices:
        i = int(idx)
        top_predictions.append({
            "class_name": class_map[i],
            "confidence": round(float(probs[i]), 4),
            "percentage": round(float(probs[i]) * 100, 2),
        })

    # Second prediction for margin check
    second_conf = float(probs[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
    confidence_margin = top_conf - second_conf

    # Uncertainty handling
    is_uncertain = False
    uncertainty_reasons = []

    if top_conf < conf_threshold:
        is_uncertain = True
        uncertainty_reasons.append(
            f"Confidence ({top_conf*100:.1f}%) is below the reliable diagnostic threshold ({conf_threshold*100:.0f}%)."
        )

    if confidence_margin < margin_threshold:
        is_uncertain = True
        alt_name = class_map[int(sorted_indices[1])]
        uncertainty_reasons.append(
            f"Close margin ({confidence_margin*100:.1f}%) with alternative candidate '{alt_name}' ({second_conf*100:.1f}%)."
        )

    # Advisory
    advisory = ADVISORY_GUIDANCE.get(top_class, {
        "status": "Informational",
        "description": "Diagnostic result recorded. Follow standard crop monitoring protocols.",
        "cultural_measures": ["Inspect plant canopy for pest or nutritional anomalies."],
        "monitoring": "Standard periodic scouting.",
    })

    return {
        "predicted_class": top_class,
        "confidence": round(top_conf, 4),
        "confidence_pct": round(top_conf * 100, 1),
        "confidence_margin": round(confidence_margin, 4),
        "is_uncertain": is_uncertain,
        "uncertainty_reasons": uncertainty_reasons,
        "top_predictions": top_predictions,
        "advisory": advisory,
        "supported_classes": list(class_map.values()),
        "model_version": "1.0.0-experimental (YOLOv8n-cls)",
        "disclaimer": SAFETY_DISCLAIMER,
        "error": None,
    }
