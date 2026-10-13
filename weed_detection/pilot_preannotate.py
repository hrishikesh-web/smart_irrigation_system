"""
PILOT: pre-annotate 3 of your own "mix tulsi and weed" photos with the EXISTING detector.

Purpose: see whether the current model's boxes are a useful starting point for labelling your
Tulsi photos. Everything this script writes is an UNVERIFIED DRAFT made by a model that was
trained on a different dataset. It is NOT ground truth and must be checked box by box.

What it does
  1. Loads weed_detection/models/weed_detector.pt through the existing detector code and prints
     the class mapping read from the model itself (must be 0 = crop, 1 = weed).
  2. Picks 3 photos (alphabetically first, or the ones you name with --images).
  3. Runs the model with a LOW floor (0.05) so weak guesses are visible, then splits them:
       - confidence >= --draft-threshold (default 0.25): written to the draft YOLO label file
       - below that: listed in the report and drawn dashed in the preview, NOT written as labels
     If an image has no box above the threshold, NO label file is written for it
     (an empty label file would wrongly claim "this photo contains no plants").
  4. Writes everything to a new folder  weed_detection/pilot_preannotations/run_<date-time>/
     Your photos, existing labels, the model and the dashboard are never modified.

It does not train anything, relabel anything, or process all 93 photos.

Run from the project root (with .venv active):
    python weed_detection\\pilot_preannotate.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for the package import
from weed_detection import detector, weed_settings as cfg  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SOURCE = Path(r"C:\Users\Hrishikesh Khanna\Downloads\project tulsi dataset photos\mix tulsi and weed")
OUT_ROOT = cfg.WEED_DIR / "pilot_preannotations"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
LOW_FLOOR = 0.05                      # lowest confidence recorded at all
COLOURS = {"crop": (40, 130, 255), "weed": (235, 45, 45)}
BANNER = "UNVERIFIED DRAFT - model predictions, NOT ground truth"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dashed_rect(draw, box, colour, width, dash=14):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min(1.0, (i + 1) / steps)
            draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1),
                      fill=colour, width=width)


def draw_preview(img: Image.Image, rows: list[dict], threshold: float, title: str) -> Image.Image:
    im = img.copy()
    W, H = im.size
    lw = max(3, W // 250)
    f_box, f_banner = font(max(16, W // 45)), font(max(18, W // 38))
    d = ImageDraw.Draw(im)
    for r in rows:
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        label = f"{r['class_name']} {r['confidence']:.2f}"
        if r["in_draft"]:
            colour = COLOURS[r["class_name"]]
            d.rectangle(box, outline=colour, width=lw)
        else:
            colour = (170, 170, 170)
            dashed_rect(d, box, colour, max(2, lw // 2))
            label += " (low, not in draft)"
        tb = d.textbbox((0, 0), label, font=f_box)
        ty = max(0, r["y1"] - (tb[3] - tb[1]) - 6)
        d.rectangle((r["x1"], ty, r["x1"] + tb[2] - tb[0] + 8, ty + tb[3] - tb[1] + 6), fill=colour)
        d.text((r["x1"] + 4, ty + 2), label, fill=(255, 255, 255), font=f_box)
    # banner
    bh = int(H * 0.075) + 10
    canvas = Image.new("RGB", (W, H + bh), (25, 25, 25))
    canvas.paste(im, (0, bh))
    d2 = ImageDraw.Draw(canvas)
    n_draft = sum(r["in_draft"] for r in rows)
    d2.text((12, 6), BANNER, fill=(255, 210, 60), font=f_banner)
    d2.text((12, 6 + bh // 2), f"{title} | {n_draft} box(es) >= {threshold:.2f} in draft, "
            f"{len(rows) - n_draft} lower-confidence shown dashed", fill=(230, 230, 230), font=f_box)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description="Pilot: draft pre-annotations for 3 of your photos (unverified).")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="folder with your mixed photos")
    ap.add_argument("--count", type=int, default=3, help="how many photos (pilot limit: max 3)")
    ap.add_argument("--images", nargs="*", default=None, help="optional exact file names to use instead")
    ap.add_argument("--draft-threshold", type=float, default=cfg.DEFAULT_CONFIDENCE,
                    help="boxes at or above this confidence go into the draft label files (default %(default)s)")
    args = ap.parse_args()

    if args.count > 3:
        print("[STOP] This is the 3-photo pilot. Use --count 3 or fewer.")
        return 1
    if not LOW_FLOOR < args.draft_threshold < 1:
        print(f"[STOP] --draft-threshold must be between {LOW_FLOOR} and 1.")
        return 1
    src = args.source
    if not src.is_dir():
        print(f"[STOP] Photo folder not found:\n  {src}\nPass the right one with --source \"<folder>\"")
        return 1

    # ---- 1. model + class mapping, confirmed from the model file itself ----
    print("1. Loading the existing detector")
    try:
        model = detector.load_model(cfg.FINAL_MODEL)   # also refuses a model whose classes differ
    except detector.WeedModelError as exc:
        print(f"[STOP] {exc}")
        return 1
    model_names = {int(k): v for k, v in model.names.items()}
    name_to_id = {v: k for k, v in model_names.items()}
    print(f"   model file      : {cfg.FINAL_MODEL}")
    print(f"   classes in model: {model_names}")
    print(f"   repository code : {cfg.EXPECTED_NAMES}  (weed_settings.EXPECTED_NAMES)")
    if model_names != cfg.EXPECTED_NAMES:
        print("[STOP] Class mapping differs from 0 = crop, 1 = weed. Nothing was written.")
        return 1

    # ---- 2. choose the photos ----
    all_imgs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if args.images:
        chosen = [src / n for n in args.images]
        missing = [p.name for p in chosen if not p.is_file()]
        if missing:
            print(f"[STOP] Not found in {src}: {missing}")
            return 1
        chosen = chosen[:3]
    else:
        chosen = all_imgs[: args.count]
    if not chosen:
        print(f"[STOP] No JPG/PNG photos in {src}")
        return 1
    print(f"2. Using {len(chosen)} of {len(all_imgs)} photos in the folder: {[p.name for p in chosen]}")

    # ---- 3. output folder (always new; never inside the photo folder) ----
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = OUT_ROOT / f"run_{stamp}"
    if src.resolve() in out.resolve().parents or out.resolve() == src.resolve():
        print("[STOP] Output folder would be inside the photo folder.")
        return 1
    labels_dir = out / "draft_labels_UNVERIFIED"
    previews_dir = out / "previews"
    labels_dir.mkdir(parents=True, exist_ok=False)
    previews_dir.mkdir()
    before = {p: sha256_of(p) for p in chosen}

    # ---- 4. predict ----
    print(f"3. Predicting (records everything >= {LOW_FLOOR}, drafts only >= {args.draft_threshold})")
    rows_all, per_image, previews = [], [], []
    for p in chosen:
        info = {"image": p.name, "status": None, "boxes_recorded": 0, "boxes_in_draft": 0,
                "draft_label_file": None, "exif_orientation": None}
        try:
            with Image.open(p) as raw:
                info["exif_orientation"] = int(raw.getexif().get(0x0112, 1))
                info["stored_size"] = list(raw.size)
            data = p.read_bytes()
            res = detector.detect(model, data, conf=LOW_FLOOR)
            upright = detector.read_image(data)
        except Exception as exc:  # noqa: BLE001 - report and continue with the other photos
            info["status"] = f"SKIPPED: {exc}"
            per_image.append(info)
            print(f"   {p.name}: {info['status']}")
            continue
        W, H = res["image_size"]
        info["upright_size"] = [W, H]
        rows = []
        for det in res["detections"]:
            x1, y1, x2, y2 = det["box_xyxy"]
            in_draft = det["confidence"] >= args.draft_threshold
            rows.append({
                "image": p.name, "class_id": name_to_id[det["class"]], "class_name": det["class"],
                "confidence": det["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "yolo_cx": round((x1 + x2) / 2 / W, 6), "yolo_cy": round((y1 + y2) / 2 / H, 6),
                "yolo_w": round((x2 - x1) / W, 6), "yolo_h": round((y2 - y1) / H, 6),
                "in_draft": in_draft,
                "note": "draft (unverified)" if in_draft else f"below {args.draft_threshold}, not written",
            })
        info["boxes_recorded"] = len(rows)
        info["boxes_in_draft"] = sum(r["in_draft"] for r in rows)
        info["crop_in_draft"] = sum(r["in_draft"] and r["class_name"] == "crop" for r in rows)
        info["weed_in_draft"] = sum(r["in_draft"] and r["class_name"] == "weed" for r in rows)
        if info["boxes_in_draft"]:
            lbl = labels_dir / f"{p.stem}.txt"
            lbl.write_text("".join(f"{r['class_id']} {r['yolo_cx']} {r['yolo_cy']} {r['yolo_w']} {r['yolo_h']}\n"
                                   for r in rows if r["in_draft"]), encoding="utf-8")
            info["draft_label_file"] = lbl.name
            info["status"] = "draft written (unverified)"
        elif rows:
            info["status"] = "only low-confidence guesses - no draft label written"
        else:
            info["status"] = f"no detections at all (even >= {LOW_FLOOR}) - no draft label written"
        rows_all += rows
        per_image.append(info)
        prev = draw_preview(upright, rows, args.draft_threshold, p.name)
        prev.thumbnail((1400, 1400))
        prev.save(previews_dir / f"{p.stem}_preview.jpg", quality=90)
        previews.append(prev)
        print(f"   {p.name}: {info['status']}; {info['crop_in_draft']} crop + {info['weed_in_draft']} weed in draft, "
              f"{info['boxes_recorded'] - info['boxes_in_draft']} low-confidence")
        if info["exif_orientation"] not in (1, None):
            print(f"      note: this photo is stored rotated (EXIF orientation {info['exif_orientation']}); "
                  "the boxes refer to the upright view shown in the preview.")

    # ---- 5. combined preview (side by side) ----
    if previews:
        h = 900
        tiles = [im.resize((int(im.width * h / im.height), h)) for im in previews]
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), h), (25, 25, 25))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width + 10
        sheet.save(out / "PILOT_PREVIEW.jpg", quality=90)

    # ---- 6. report files ----
    fields = ["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
              "yolo_cx", "yolo_cy", "yolo_w", "yolo_h", "in_draft", "note"]
    with open(out / "predictions_UNVERIFIED.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_all)
    unchanged = all(sha256_of(p) == h for p, h in before.items())
    summary = {
        "status": "UNVERIFIED DRAFT - model predictions, not ground truth",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": str(cfg.FINAL_MODEL), "model_sha256": sha256_of(cfg.FINAL_MODEL),
        "class_mapping_from_model": model_names, "class_mapping_in_repository": cfg.EXPECTED_NAMES,
        "source_folder": str(src), "photos_in_folder": len(all_imgs),
        "low_confidence_floor": LOW_FLOOR, "draft_threshold": args.draft_threshold,
        "coordinates": "pixels of the upright image (EXIF rotation applied); YOLO values normalised to it",
        "images": per_image, "original_photos_unchanged": unchanged,
    }
    (out / "summary_UNVERIFIED.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README_UNVERIFIED.txt").write_text(
        "THESE ARE UNVERIFIED DRAFTS.\n\n"
        "The boxes were predicted by weed_detector.pt, which was trained on a different crop-and-weed\n"
        "dataset (not Tulsi). They are NOT ground truth. Do not train on them without checking and\n"
        "correcting every box by hand.\n\n"
        "draft_labels_UNVERIFIED/  YOLO .txt files (class cx cy w h), only boxes >= the draft threshold.\n"
        "                          No file = no box reached the threshold (NOT 'no plants in photo').\n"
        "predictions_UNVERIFIED.csv  every prediction >= 0.05 with pixel and YOLO coordinates.\n"
        "previews/ and PILOT_PREVIEW.jpg  solid boxes = in draft, dashed grey = low confidence.\n"
        "summary_UNVERIFIED.json  settings, class mapping read from the model, per-image status.\n"
        "Classes: 0 = crop (Tulsi), 1 = weed.\n", encoding="utf-8")

    print(f"4. Original photos unchanged: {'YES' if unchanged else 'NO - investigate'}")
    print(f"\nDone. Open this preview to judge the drafts:\n  {out / 'PILOT_PREVIEW.jpg'}")
    print(f"All outputs: {out}")
    print("Reminder: these are UNVERIFIED model guesses, not ground truth.")
    return 0 if unchanged else 2


if __name__ == "__main__":
    sys.exit(main())
