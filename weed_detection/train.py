"""
STAGE 2 - Fine-tune an official pretrained YOLO model on the prepared crop/weed dataset.

What happens when you run it
    1. Pre-flight checks, before anything heavy:
         - data.yaml exists, says 0 = crop / 1 = weed, and every image has a label
         - the Stage 1 leakage check passed
         - PyTorch can see your NVIDIA GPU (stops with instructions if not)
         - Ultralytics is installed
    2. Downloads the official pretrained checkpoint (yolo11n.pt by default) from
       Ultralytics' GitHub releases into weed_detection/pretrained/ and checks its
       SHA-256 fingerprint. No weights from the old repository are used.
    3. Fine-tunes it. Every run gets its own folder:  weed_detection/runs/train_<name>/
    4. Copies the best checkpoint to  weed_detection/models/weed_detector.pt
       (an older one is moved to models/archive/ first, never silently overwritten)
       and writes weed_detector_info.json describing exactly how it was made.

It does NOT evaluate on the test split. Run evaluate.py for that afterwards.

Usage (PowerShell, from the project root, with .venv active):
    python weed_detection\\train.py --check-only          # pre-flight checks only, no training
    python weed_detection\\train.py                        # full training with default settings
    python weed_detection\\train.py --epochs 50 --batch 8  # override settings
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

import weed_settings as cfg


class PreflightError(SystemExit):
    """Stops the script with a readable message and exit code 1."""

    def __init__(self, message: str):
        print("\n[STOP] " + message.strip() + "\n")
        super().__init__(1)


# ---------------------------------------------------------------------------- dataset checks

def _read_simple_yaml(path: Path) -> dict:
    """Read data.yaml without extra libraries (it is a very simple file we generated)."""
    data, names, in_names = {}, {}, False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if in_names and raw.startswith((" ", "\t")):
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip().strip("'\"")
            continue
        in_names = False
        k, v = line.split(":", 1)
        if k.strip() == "names" and not v.strip():
            in_names = True
        else:
            data[k.strip()] = v.strip().strip("'\"")
    data["names"] = names
    return data


def check_dataset(data_yaml: Path) -> tuple[Path, dict]:
    """Validate the prepared dataset. Returns (dataset_root, image counts per split)."""
    if not data_yaml.is_file():
        raise PreflightError(
            f"Prepared dataset not found: {data_yaml}\n"
            "Run Stage 1 first:  python weed_detection\\prepare_dataset.py --source \"<weed_dataset folder>\"")
    y = _read_simple_yaml(data_yaml)
    if y["names"] != cfg.EXPECTED_NAMES:
        raise PreflightError(f"data.yaml class names are {y['names']}, expected {cfg.EXPECTED_NAMES}.")

    # Always use the folder that data.yaml actually sits in, so the project can be moved
    # (or uploaded to Colab) without editing the absolute 'path:' line written in Stage 1.
    root = data_yaml.parent.resolve()
    written = Path(y.get("path", str(root)))
    if written.resolve() != root:
        print(f"  note: data.yaml 'path' points to {written}; using the real location {root} instead.")

    counts = {}
    for split in ("train", "val", "test"):
        img_dir = root / y.get(split, f"images/{split}")
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise PreflightError(f"Missing folder for split '{split}': {img_dir} or {lbl_dir}")
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        if not imgs:
            raise PreflightError(f"No images in {img_dir}")
        if imgs != lbls:
            raise PreflightError(f"Split '{split}': {len(imgs ^ lbls)} images/labels do not match up. "
                                 "Rebuild with prepare_dataset.py --overwrite.")
        counts[split] = len(imgs)

    if cfg.SPLIT_REPORT.is_file():
        rep = json.loads(cfg.SPLIT_REPORT.read_text(encoding="utf-8"))
        if not rep.get("leakage_check", {}).get("passed", False):
            raise PreflightError("Stage 1 leakage check did not pass (see reports/split_report.json).")
    else:
        print("  warning: reports/split_report.json not found, so the leakage check cannot be confirmed.")
    return root, counts


def write_resolved_yaml(dataset_root: Path, dest: Path) -> Path:
    """A copy of data.yaml whose 'path' is the dataset's real current location."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = (f"# Auto-generated by train.py from {dataset_root / 'data.yaml'}\n"
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "".join(f"  {k}: {v}\n" for k, v in sorted(cfg.EXPECTED_NAMES.items())))
    dest.write_text(text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------- environment checks

def check_environment(allow_cpu: bool) -> dict:
    try:
        import torch
    except ImportError:
        raise PreflightError("PyTorch is not installed in this Python environment.\n"
                             "Activate the project's .venv first:  .\\.venv\\Scripts\\Activate.ps1")
    try:
        import ultralytics
    except ImportError:
        raise PreflightError("Ultralytics is not installed. Run:  python -m pip install ultralytics")

    info = {"python": platform.python_version(), "os": platform.platform(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["device"] = "0"
    elif allow_cpu:
        print("  WARNING: no CUDA GPU - running on CPU because --allow-cpu was given (very slow).")
        info["device"] = "cpu"
    else:
        cuda_build = torch.version.cuda
        raise PreflightError(
            "PyTorch cannot see a CUDA GPU, so training was NOT started.\n"
            f"  installed torch: {torch.__version__} (CUDA build: {cuda_build or 'none - this is the CPU-only build'})\n"
            "How to fix:\n"
            "  1. Run  nvidia-smi  in PowerShell. If it fails, update the NVIDIA driver and restart.\n"
            "  2. If torch says 'CPU-only build', reinstall the CUDA build inside .venv:\n"
            "       python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126\n"
            "  3. Check:  python -c \"import torch; print(torch.cuda.is_available())\"   (must print True)\n"
            "Or train on Google Colab instead (see README_STAGE2.md).")
    return info


# ---------------------------------------------------------------------------- official checkpoint

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_official_checkpoint(name: str) -> Path:
    spec = cfg.OFFICIAL_CHECKPOINTS[name]
    dest = cfg.PRETRAINED_DIR / f"{name}.pt"
    cfg.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_of(dest) == spec["sha256"]:
        print(f"  official checkpoint already downloaded and verified: {dest}")
        return dest
    print(f"  downloading {spec['url']}")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(spec["url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise PreflightError(
            f"Could not download the official checkpoint: {exc}\n"
            f"Download it in your browser from:\n  {spec['url']}\n"
            f"and save it as:\n  {dest}\nthen run this command again.")
    got = sha256_of(tmp)
    if got != spec["sha256"]:
        tmp.unlink(missing_ok=True)
        raise PreflightError(f"Downloaded file fingerprint mismatch.\n expected {spec['sha256']}\n got      {got}")
    tmp.replace(dest)
    print(f"  saved and verified (SHA-256 OK): {dest}")
    return dest


# ---------------------------------------------------------------------------- main

def main() -> int:
    d = cfg.TRAIN_DEFAULTS
    ap = argparse.ArgumentParser(description="Fine-tune an official YOLO checkpoint on the crop/weed dataset.")
    ap.add_argument("--model", choices=sorted(cfg.OFFICIAL_CHECKPOINTS), default=cfg.DEFAULT_CHECKPOINT,
                    help="official starting checkpoint (default %(default)s)")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--patience", type=int, default=d["patience"])
    ap.add_argument("--imgsz", type=int, default=d["imgsz"])
    ap.add_argument("--batch", type=int, default=d["batch"])
    ap.add_argument("--workers", type=int, default=d["workers"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--name", default=None, help="run folder name (default: <model>_<imgsz>_<date-time>)")
    ap.add_argument("--data", type=Path, default=cfg.DATA_YAML)
    ap.add_argument("--check-only", action="store_true", help="run pre-flight checks and the download, then stop")
    ap.add_argument("--allow-cpu", action="store_true", help="testing only: allow (very slow) CPU training")
    ap.add_argument("--no-install-model", action="store_true",
                    help="do not copy best.pt to models/weed_detector.pt after training")
    args = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.name or f"{args.model}_{args.imgsz}_{stamp}"

    print("== 1/4 Checking prepared dataset")
    dataset_root, counts = check_dataset(args.data)
    print(f"  dataset: {dataset_root}\n  images: {counts}")

    print("== 2/4 Checking Python / PyTorch / GPU")
    env = check_environment(args.allow_cpu)
    for k, v in env.items():
        print(f"  {k:17s}: {v}")

    print("== 3/4 Getting the official pretrained checkpoint")
    ckpt = get_official_checkpoint(args.model)

    settings = {"model": args.model, "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
                "batch": args.batch, "workers": args.workers, "seed": args.seed, "device": env["device"]}
    print("== Settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()))

    if args.check_only:
        print("\n--check-only given: all checks passed. No training was started.")
        return 0

    print(f"== 4/4 Training. Output folder: {cfg.RUNS_DIR / ('train_' + run_name)}")
    from ultralytics import YOLO

    resolved_yaml = write_resolved_yaml(dataset_root, cfg.RUNS_DIR / "_resolved_data.yaml")
    model = YOLO(str(ckpt))
    model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=env["device"],
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        project=str(cfg.RUNS_DIR.resolve()),  # absolute, so outputs stay inside weed_detection/runs
        name=f"train_{run_name}",
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[STOP] Training finished but no best.pt was found in {save_dir / 'weights'}.")
        return 1

    info = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(save_dir),
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256_of(best),
        "starting_checkpoint": {"name": args.model, **cfg.OFFICIAL_CHECKPOINTS[args.model]},
        "dataset": str(dataset_root), "images_per_split": counts,
        "class_names": cfg.EXPECTED_NAMES, "settings": settings, "environment": env,
        "note": "Validation metrics from training are in run_dir/results.csv. "
                "Held-out TEST metrics come only from evaluate.py.",
    }
    (save_dir / "weed_detector_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if not args.no_install_model:
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if cfg.FINAL_MODEL.exists():
            archive = cfg.MODELS_DIR / "archive"
            archive.mkdir(exist_ok=True)
            old = archive / f"weed_detector_before_{stamp}.pt"
            shutil.move(str(cfg.FINAL_MODEL), old)
            if cfg.FINAL_MODEL_INFO.exists():
                shutil.move(str(cfg.FINAL_MODEL_INFO), archive / f"weed_detector_info_before_{stamp}.json")
            print(f"  previous model moved to {old}")
        shutil.copy2(best, cfg.FINAL_MODEL)
        cfg.FINAL_MODEL_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n================ TRAINING FINISHED ================")
    print(f"Run folder        : {save_dir}")
    print(f"Best checkpoint   : {best}")
    if not args.no_install_model:
        print(f"Installed model   : {cfg.FINAL_MODEL}")
    print("Training curves   : results.png and results.csv in the run folder (these are VALIDATION numbers).")
    print("Next step - measure on the held-out TEST images:")
    print("  python weed_detection\\evaluate.py")
    return 0


if __name__ == "__main__":  # required on Windows: data-loader workers re-import this file
    sys.exit(main())
