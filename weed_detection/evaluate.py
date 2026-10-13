"""
STAGE 2 - Measure the trained model on the HELD-OUT TEST images.

The test split (190 images, 41 independent photo groups) was never used for training or for
choosing the best checkpoint, so these are the honest numbers to report.

It produces two kinds of results:
  A. Standard object-detection metrics from Ultralytics (precision, recall, mAP50, mAP50-95),
     overall and per class. Here a box only counts if it overlaps the true box well.
  B. A simple image-level check that matches how the dashboard will be used:
     "Does the model report at least one weed in this image?" compared with the true labels,
     at the confidence threshold the dashboard will use (default 0.25).

Outputs
  weed_detection/runs/eval_<split>_<date-time>/        Ultralytics plots + example predictions
  weed_detection/reports/<split>_metrics.json          every number, machine-readable
  weed_detection/reports/<split>_metrics.md            the same numbers as a readable table

Usage (PowerShell, from the project root, with .venv active):
    python weed_detection\\evaluate.py                       # test split, models\\weed_detector.pt
    python weed_detection\\evaluate.py --conf 0.35           # different dashboard threshold
    python weed_detection\\evaluate.py --weights weed_detection\\runs\\train_xxx\\weights\\best.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import weed_settings as cfg
from train import PreflightError, check_dataset, check_environment, sha256_of, write_resolved_yaml

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def refuse_untrusted(weights: Path) -> None:
    """Only accept models produced by this new pipeline (inside weed_detection/)."""
    w = weights.resolve()
    if cfg.WEED_DIR.resolve() not in w.parents:
        raise PreflightError(f"{w} is outside weed_detection/. Only models trained by weed_detection/train.py "
                             "are evaluated, so old repository weights cannot be used by mistake.")
    if w.parent.resolve() == cfg.PRETRAINED_DIR.resolve():
        raise PreflightError("That is the untrained official starting checkpoint (it only knows COCO classes). "
                             "Evaluate the fine-tuned model instead.")


def image_level_check(model, img_dir: Path, lbl_dir: Path, conf: float, imgsz: int, device: str,
                      save_examples_to: Path, n_examples: int = 12) -> dict:
    weed_id = next(k for k, v in cfg.EXPECTED_NAMES.items() if v == "weed")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    tp = fp = fn = tn = 0
    misses, false_alarms = [], []
    save_examples_to.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(model.predict(source=[str(p) for p in images], conf=conf, imgsz=imgsz,
                                        device=device, stream=True, verbose=False)):
        p = images[i]
        lines = (lbl_dir / f"{p.stem}.txt").read_text(encoding="utf-8").split("\n")
        truth = any(ln.split()[0] == str(weed_id) for ln in lines if ln.strip())
        pred = bool(r.boxes is not None and len(r.boxes) and (r.boxes.cls == weed_id).any().item())
        if truth and pred:
            tp += 1
        elif truth:
            fn += 1
            misses.append(p.stem)
        elif pred:
            fp += 1
            false_alarms.append(p.stem)
        else:
            tn += 1
        if i < n_examples:
            r.save(filename=str(save_examples_to / f"{p.stem}_pred.jpg"))

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {
        "confidence_threshold": conf,
        "images": len(images),
        "images_with_weed_in_labels": tp + fn,
        "images_without_weed_in_labels": fp + tn,
        "weed_found_correctly (TP)": tp,
        "weed_missed (FN)": fn,
        "false_weed_alarm (FP)": fp,
        "correctly_no_weed (TN)": tn,
        "accuracy": ratio(tp + tn, len(images)),
        "weed_recall (share of weedy images flagged)": ratio(tp, tp + fn),
        "weed_precision (share of flags that were right)": ratio(tp, tp + fp),
        "missed_images": misses,
        "false_alarm_images": false_alarms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the trained weed model on the held-out test split.")
    ap.add_argument("--weights", type=Path, default=cfg.FINAL_MODEL)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--conf", type=float, default=cfg.DEFAULT_CONFIDENCE,
                    help="confidence threshold for the image-level check (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=cfg.TRAIN_DEFAULTS["imgsz"])
    ap.add_argument("--batch", type=int, default=cfg.TRAIN_DEFAULTS["batch"])
    ap.add_argument("--data", type=Path, default=cfg.DATA_YAML)
    ap.add_argument("--allow-cpu", action="store_true", help="evaluate without a GPU (slower, same numbers)")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise PreflightError(f"Model not found: {args.weights}\nTrain first:  python weed_detection\\train.py")
    refuse_untrusted(args.weights)
    dataset_root, counts = check_dataset(args.data)
    env = check_environment(args.allow_cpu)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = {int(k): v for k, v in model.names.items()}
    if names != cfg.EXPECTED_NAMES:
        raise PreflightError(f"Model classes are {names}, expected {cfg.EXPECTED_NAMES}. Wrong model file?")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"eval_{args.split}_{stamp}"
    resolved_yaml = write_resolved_yaml(dataset_root, cfg.RUNS_DIR / "_resolved_data.yaml")

    print(f"== A. Detection metrics on the {args.split.upper()} split ({counts[args.split]} images)")
    m = model.val(data=str(resolved_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=env["device"], project=str(cfg.RUNS_DIR.resolve()), name=run_name,
                  exist_ok=False, plots=True, verbose=True)
    overall = {k.replace("metrics/", "").replace("(B)", ""): round(float(v), 4)
               for k, v in m.results_dict.items() if k != "fitness"}
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                    "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}

    print(f"== B. Image-level 'is there a weed?' check at confidence {args.conf}")
    img_dir = dataset_root / "images" / args.split
    lbl_dir = dataset_root / "labels" / args.split
    save_dir = Path(m.save_dir)
    image_level = image_level_check(model, img_dir, lbl_dir, args.conf, args.imgsz, env["device"],
                                    save_dir / "example_predictions")

    result = {
        "evaluated": _dt.datetime.now().isoformat(timespec="seconds"),
        "split": args.split, "images": counts[args.split],
        "weights": str(args.weights.resolve()), "weights_sha256": sha256_of(args.weights),
        "eval_run_dir": str(save_dir), "imgsz": args.imgsz,
        "detection_metrics_overall": overall, "detection_metrics_per_class": per_class,
        "image_level_weed_check": image_level, "environment": env,
    }
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = cfg.REPORTS_DIR / f"{args.split}_metrics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [f"# Weed detector - {args.split} split results", "",
          f"Evaluated {result['evaluated']} on {counts[args.split]} held-out images.",
          f"Model: `{args.weights}` (SHA-256 `{result['weights_sha256'][:16]}...`)", "",
          "## A. Detection metrics (Ultralytics)", "",
          "| Class | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|",
          f"| **all** | {overall.get('precision')} | {overall.get('recall')} | {overall.get('mAP50')} | "
          f"{overall.get('mAP50-95')} |"]
    md += [f"| {c} | {v['precision']} | {v['recall']} | {v['mAP50']} | {v['mAP50-95']} |" for c, v in per_class.items()]
    il = image_level
    md += ["", f"## B. Image-level weed check (confidence >= {args.conf})", "",
           "| | Model says weed | Model says no weed |", "|---|---|---|",
           f"| Image really has weed | {il['weed_found_correctly (TP)']} | {il['weed_missed (FN)']} |",
           f"| Image has no weed | {il['false_weed_alarm (FP)']} | {il['correctly_no_weed (TN)']} |", "",
           f"Accuracy {il['accuracy']}, weed recall {il['weed_recall (share of weedy images flagged)']}, "
           f"weed precision {il['weed_precision (share of flags that were right)']}.", "",
           "These numbers describe this dataset's test photos only (one crop species, single plants on bare soil). "
           "They do not show the model works on other crops, other weed species, or dense field scenes."]
    md_path = cfg.REPORTS_DIR / f"{args.split}_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n================ EVALUATION FINISHED ================")
    print("\n".join(md[4:]))
    print(f"\nSaved: {json_path}\n       {md_path}\nPlots and example predictions: {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
