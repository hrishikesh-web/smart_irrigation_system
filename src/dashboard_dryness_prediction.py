"""
Dashboard helper for the four-feature "future dryness" Random Forest
(models/future_dryness_model.joblib, trained by src/train_model.py).

This file was missing from the shared project; it is written to match how the dashboard calls it:
    predict_future_dryness(soil_moisture_1=..., soil_moisture_2=..., soil_temperature=..., rainfall=...)
and the bundle saved by src/train_model.py:
    {"model", "features", "forecast_minutes", "low_moisture_threshold", "label_definition"}

It never raises: it returns {"available": False, "message": ...} when something is wrong, so the
dashboard can show the reason instead of crashing. It does not read sensors or control the pump.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "future_dryness_model.joblib"
REQUIRED_KEYS = ("model", "features", "forecast_minutes", "low_moisture_threshold")


@lru_cache(maxsize=1)
def _load_bundle(path_str: str, modified: float):
    """Load once per file version (the modified time is part of the cache key)."""
    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(path_str)
    version_warnings = [str(w.message) for w in caught if "version" in str(w.message).lower()]
    return bundle, version_warnings


def model_info() -> dict:
    """Cheap status check for the dashboard."""
    exists = MODEL_PATH.is_file()
    return {"path": str(MODEL_PATH), "exists": exists,
            "size_kb": round(MODEL_PATH.stat().st_size / 1024) if exists else None}


def predict_future_dryness(soil_moisture_1, soil_moisture_2, soil_temperature, rainfall) -> dict:
    """Predict whether soil moisture ~30 minutes ahead will be at/below the training threshold."""
    if not MODEL_PATH.is_file():
        return {"available": False,
                "message": f"Model file not found: {MODEL_PATH}. Create it with:  python src\\train_model.py"}
    try:
        import pandas as pd
        bundle, version_warnings = _load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        return {"available": False, "message": f"Could not load the model file: {exc}"}

    if not isinstance(bundle, dict) or any(k not in bundle for k in REQUIRED_KEYS):
        return {"available": False,
                "message": "Model file has an unexpected format. Re-create it with:  python src\\train_model.py"}

    values = {
        "soil_moisture_1": soil_moisture_1,
        "soil_moisture_2": soil_moisture_2,
        "soil_temperature": soil_temperature,
        "rainfall": rainfall,
    }
    try:
        sample = pd.DataFrame([{f: float(values[f]) for f in bundle["features"]}], columns=bundle["features"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"available": False, "message": f"Invalid input for the model's features: {exc}"}

    try:
        model = bundle["model"]
        prediction = int(model.predict(sample)[0])
        probability = float(model.predict_proba(sample)[0][1])
    except Exception as exc:
        hint = ""
        if version_warnings:
            hint = (" The file was saved with a different scikit-learn version than the one installed."
                    " Re-create it in this environment with:  python src\\train_model.py")
        return {"available": False, "message": f"Prediction failed: {exc}.{hint}"}

    return {
        "available": True,
        "prediction": prediction,               # 1 = low moisture predicted
        "probability": probability,             # model probability for the low-moisture class
        "forecast_minutes": int(bundle["forecast_minutes"]),
        "threshold": float(bundle["low_moisture_threshold"]),
        "label_definition": bundle.get("label_definition"),
        "version_warning": version_warnings[0] if version_warnings else None,
    }
