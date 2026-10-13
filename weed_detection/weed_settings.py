"""
One place for every path and setting the weed-detection pipeline uses.

train.py and evaluate.py read from here, and the dashboard will too in Stage 3,
so a model trained here is always found in the same place.
All paths are built with pathlib, so they work on Windows, macOS, Linux and Colab.
"""

from pathlib import Path

# ---------------------------------------------------------------- folders
WEED_DIR = Path(__file__).resolve().parent            # .../weed_detection
PROJECT_ROOT = WEED_DIR.parent                         # .../Smart_irrigation_system_new-main
DATA_YAML = WEED_DIR / "dataset" / "data.yaml"         # made by prepare_dataset.py in Stage 1
SPLIT_REPORT = WEED_DIR / "reports" / "split_report.json"
PRETRAINED_DIR = WEED_DIR / "pretrained"               # official starting checkpoints are downloaded here
RUNS_DIR = WEED_DIR / "runs"                           # every training / evaluation run gets a sub-folder
MODELS_DIR = WEED_DIR / "models"                       # the final model the dashboard will use
FINAL_MODEL = MODELS_DIR / "weed_detector.pt"
FINAL_MODEL_INFO = MODELS_DIR / "weed_detector_info.json"
REPORTS_DIR = WEED_DIR / "reports"

EXPECTED_NAMES = {0: "crop", 1: "weed"}

# ---------------------------------------------------------------- official checkpoints
# Downloaded directly from Ultralytics' official GitHub release assets. Nothing from the
# old repository (for example training/yolov8n.pt) is ever used.
# The SHA-256 fingerprints were recorded from these exact URLs on 2026-10-12. If a download
# does not match, training stops, so a corrupted or swapped file can never be used.
OFFICIAL_CHECKPOINTS = {
    "yolo11n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "size_bytes": 5613764,
    },
    "yolo26n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        "size_bytes": 5544453,
    },
}
DEFAULT_CHECKPOINT = "yolo11n"

# ---------------------------------------------------------------- training defaults (8 GB GPU)
TRAIN_DEFAULTS = {
    "epochs": 80,        # full passes over the training images; early stopping may end sooner
    "patience": 25,      # stop early if validation mAP has not improved for this many epochs
    "imgsz": 512,        # the dataset images are 512x512, so no upscaling is wasted
    "batch": 16,         # images per step; a nano model at 512 px fits comfortably in 8 GB
    "workers": 4,        # CPU processes that load images; lower to 2 if Windows struggles
    "seed": 42,          # fixed randomness so a re-run gives (nearly) the same result
}

# ---------------------------------------------------------------- inference default
DEFAULT_CONFIDENCE = 0.25    # boxes below this confidence are hidden (adjustable in the dashboard later)
