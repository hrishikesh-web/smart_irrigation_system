"""
STEP 1 of the weed-detection pipeline: INSPECT the dataset (read-only).

What it does
    1. Reads the classes file and confirms it says: crop (id 0), weed (id 1).
    2. Confirms every image has a matching .txt label file and vice versa.
    3. Checks every label line is valid YOLO format: "class cx cy w h", all in 0..1.
    4. Opens every image to make sure none are corrupt.
    5. Finds duplicate / near-duplicate images (including flipped and rotated copies)
       and groups them, so copies of one photo can later be kept in the same split.
    6. Finds "contradictory" groups: copies of the same photo labelled with different classes.
    7. Writes a report plus two picture sheets you can open to check things by eye.

It NEVER changes, moves or deletes anything in the original dataset.

Usage (PowerShell, from the project folder):
    python weed_detection\\check_dataset.py --source "C:\\path\\to\\weed_dataset"
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import dataset_tools as dt

HERE = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = HERE / "reports"
CLASS_COLOURS = {0: (30, 120, 255), 1: (230, 30, 30)}  # crop = blue, weed = red


def analyse_dataset(source: Path, threshold: int) -> dict:
    """Run every check and return all results in one dictionary (used by prepare_dataset.py too)."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset folder not found: {source}")

    classes_file = dt.find_classes_file(source)
    class_names = dt.read_classes(classes_file)
    image_folder = dt.find_image_folder(source)

    print(f"Classes file : {classes_file}")
    print(f"Class names  : {class_names}  (id 0 = {class_names[0] if class_names else '?'}, "
          f"id 1 = {class_names[1] if len(class_names) > 1 else '?'})")
    print(f"Image folder : {image_folder}")

    fatal = []
    if class_names != dt.EXPECTED_CLASSES:
        fatal.append(f"classes file says {class_names}, expected {dt.EXPECTED_CLASSES}. "
                     "Stop and check the class order before training.")

    print("Reading images and labels ...")
    samples, orphan_labels = dt.load_samples(image_folder, num_classes=max(len(class_names), 1))
    if orphan_labels:
        fatal.append(f"{len(orphan_labels)} label files have no image, e.g. {orphan_labels[:5]}")

    print("Fingerprinting the original files (to prove later that nothing changed) ...")
    fingerprint = {str(p.relative_to(source)).replace("\\", "/"): dt.file_md5(p)
                   for p in sorted(source.rglob("*")) if p.is_file()}

    print("Looking for exact duplicates ...")
    by_md5 = collections.defaultdict(list)
    for s in samples:
        by_md5[fingerprint[str(s.image_path.relative_to(source)).replace("\\", "/")]].append(s.stem)
    exact_dups = [v for v in by_md5.values() if len(v) > 1]

    print(f"Looking for near-duplicates (8 orientations each, threshold {threshold}) - about 30 s ...")
    dist = dt.duplicate_distance_matrix(samples)
    group_ids = dt.group_duplicates(dist, threshold)
    groups = dt.analyse_groups(samples, group_ids, class_names)

    return {
        "source": source, "classes_file": classes_file, "class_names": class_names,
        "image_folder": image_folder, "samples": samples, "orphan_labels": orphan_labels,
        "fingerprint": fingerprint, "exact_duplicates": exact_dups, "distance": dist,
        "threshold": threshold, "group_ids": group_ids, "groups": groups, "fatal": fatal,
    }


def summarise(a: dict) -> dict:
    samples, names = a["samples"], a["class_names"]
    box_counts = collections.Counter(b.class_id for s in samples for b in s.boxes)
    dominant = collections.Counter(s.dominant_class for s in samples)
    sizes = collections.Counter((s.width, s.height) for s in samples)
    areas = sorted(b.area for s in samples for b in s.boxes)
    groups = a["groups"]
    contra = [g for g in groups if g["contradictory"]]
    return {
        "class_names": {i: n for i, n in enumerate(names)},
        "num_images": len(samples),
        "num_label_files": sum(s.label_path is not None for s in samples),
        "orphan_label_files": a["orphan_labels"],
        "images_with_problems": sum(bool(s.problems) for s in samples),
        "boxes_per_class": {names[c]: n for c, n in sorted(box_counts.items())},
        "images_by_main_class": {names[c] if c is not None else "none": n for c, n in dominant.items()},
        "images_containing_both_classes": sum(len(s.class_set) > 1 for s in samples),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common()},
        "box_area_fraction": {
            "min": round(areas[0], 4) if areas else None,
            "median": round(areas[len(areas) // 2], 4) if areas else None,
            "max": round(areas[-1], 4) if areas else None,
            "tiny_boxes_under_1pct": sum(x < 0.01 for x in areas),
        },
        "exact_duplicate_groups": len(a["exact_duplicates"]),
        "duplicate_threshold": a["threshold"],
        "duplicate_groups": len(groups),
        "single_images_without_copies": sum(g["size"] == 1 for g in groups),
        "largest_groups": sorted((g["size"] for g in groups), reverse=True)[:5],
        "contradictory_groups": len(contra),
        "contradictory_groups_tied": sum(g["tie"] for g in contra),
        "images_in_contradictory_groups": sum(g["size"] for g in contra),
        "fatal_errors": a["fatal"],
    }


# ---------------------------------------------------------------------------
# Picture sheets for checking by eye
# ---------------------------------------------------------------------------

def _thumb(s: dt.Sample, size: int, caption: str) -> Image.Image:
    with Image.open(s.image_path) as im:
        im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    for b in s.boxes:
        draw.rectangle([(b.cx - b.w / 2) * W, (b.cy - b.h / 2) * H, (b.cx + b.w / 2) * W, (b.cy + b.h / 2) * H],
                       outline=CLASS_COLOURS.get(b.class_id, (255, 255, 0)), width=max(3, W // 120))
    im = im.resize((size, size))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size, 14], fill=(0, 0, 0))
    d.text((3, 1), caption, fill=(255, 255, 0))
    return im


def save_class_sheet(a: dict, out: Path, per_class: int = 8, size: int = 200) -> None:
    rng = random.Random(0)
    rows = []
    for cid, name in enumerate(a["class_names"]):
        only = [s for s in a["samples"] if s.class_set == (cid,)]
        rows.append((name, rng.sample(only, min(per_class, len(only)))))
    sheet = Image.new("RGB", (per_class * size, len(rows) * size), "white")
    for r, (name, ss) in enumerate(rows):
        for c, s in enumerate(ss):
            sheet.paste(_thumb(s, size, f"{name}: {s.stem}"), (c * size, r * size))
    sheet.save(out, quality=90)


def save_conflict_sheet(a: dict, out: Path, size: int = 170) -> None:
    by_stem = {s.stem: s for s in a["samples"]}
    contra = [g for g in a["groups"] if g["contradictory"]]
    if not contra:
        return
    cols = max(g["size"] for g in contra)
    sheet = Image.new("RGB", (cols * size, len(contra) * size), "white")
    names = a["class_names"]
    for r, g in enumerate(contra):
        for c, stem in enumerate(g["stems"]):
            s = by_stem[stem]
            dom = names[s.dominant_class] if s.dominant_class is not None else "?"
            if g["tie"]:
                verdict = "DROP(tie)"
            else:
                verdict = "keep" if dom == g["majority_class"] else "DROP"
            sheet.paste(_thumb(s, size, f"G{g['group_id']} {dom} {verdict}"), (c * size, r * size))
    sheet.save(out, quality=90)


# ---------------------------------------------------------------------------

def write_reports(a: dict, summary: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(report_dir / "duplicate_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "size", "contradictory", "tie", "majority_class", "votes", "images"])
        for g in a["groups"]:
            w.writerow([g["group_id"], g["size"], g["contradictory"], g["tie"], g["majority_class"],
                        json.dumps(g["votes"]), " ".join(g["stems"])])

    with open(report_dir / "label_problems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "problem"])
        for s in a["samples"]:
            for p in s.problems:
                w.writerow([s.stem, p])

    with open(report_dir / "source_fingerprint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "md5"])
        w.writerows(sorted(a["fingerprint"].items()))

    save_class_sheet(a, report_dir / "class_samples.jpg")
    save_conflict_sheet(a, report_dir / "contradictory_groups.jpg")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only check of the crop/weed dataset.")
    ap.add_argument("--source", required=True, type=Path,
                    help="Folder that contains classes.txt and agri_data (the unzipped weed_dataset).")
    ap.add_argument("--threshold", type=int, default=dt.DEFAULT_DUPLICATE_THRESHOLD,
                    help="Near-duplicate threshold in bits out of 256 (default %(default)s).")
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    a = analyse_dataset(args.source, args.threshold)
    summary = summarise(a)
    write_reports(a, summary, args.report_dir)

    print("\n================ DATASET CHECK SUMMARY ================")
    for k, v in summary.items():
        print(f"{k:32s}: {v}")
    print("=======================================================")
    print(f"Reports written to: {args.report_dir.resolve()}")
    print("  class_samples.jpg        - blue boxes = crop, red boxes = weed. Check they look right.")
    print("  contradictory_groups.jpg - copies of one photo with different labels (keep / DROP shown).")
    print("  duplicate_groups.csv     - which images are copies of each other.")
    print("  label_problems.csv       - any bad label lines (should be empty).")
    if summary["fatal_errors"]:
        print("\nFATAL PROBLEMS - fix these before preparing the dataset:")
        for e in summary["fatal_errors"]:
            print("  -", e)
        return 1
    print("\nResult: OK - no blocking problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
