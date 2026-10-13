"""
Build a train/val/test COPY of your 93 hand-annotated Tulsi/weed photos for YOLO training.

Reads (never modifies):
  --photos-root   folder containing:  mix tulsi and weed / tulsi only / weed only
  --labels-root   folder containing LabelImg's YOLO .txt files, one sub-folder per photo folder
                  (sub-folder names may use spaces or underscores, e.g. mix_tulsi_and_weed).
                  If a label is not found there, the script also looks next to the photo.

Checks before copying anything:
  - every classes.txt it finds says exactly: crop, weed   (0 = crop, 1 = weed)
  - every photo has a label file, every label line is valid YOLO (class 0/1, values 0..1)
  - images open correctly

Splits ~70/15/15 per source folder (so each split has mix, tulsi-only and weed-only photos), keeps
near-identical shots (e.g. burst photos) in the same split, fixed seed for repeatability.

Writes:  weed_detection/tulsi_dataset/  images/{train,val,test}  labels/{train,val,test}  data.yaml
         weed_detection/tulsi_dataset/split_manifest.csv and split_report.json

Usage (PowerShell, project root, .venv active):
  python weed_detection\\prepare_tulsi_dataset.py --photos-root "<photos>" --labels-root "<labels>"
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_tools as dt  # noqa: E402  (same helpers used for the original dataset)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "tulsi_dataset"
MARKER = ".created_by_prepare_tulsi_dataset"
SPLITS = ("train", "val", "test")
DUP_THRESHOLD = 40   # stricter than the 56 used for the augmented public dataset: only near-identical shots


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos-root", type=Path, required=True)
    ap.add_argument("--labels-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    problems = []
    if not a.photos_root.is_dir():
        print(f"[STOP] Photos folder not found: {a.photos_root}")
        return 1
    photo_dirs = sorted(d for d in a.photos_root.iterdir() if d.is_dir())
    label_dirs = {norm(d.name): d for d in a.labels_root.iterdir() if d.is_dir()} if a.labels_root.is_dir() else {}
    print(f"Photo folders: {[d.name for d in photo_dirs]}")

    # classes.txt files written by LabelImg must keep crop=0, weed=1
    for cf in list(a.labels_root.rglob("classes.txt")) + list(a.photos_root.rglob("classes.txt")):
        names = dt.read_classes(cf)
        if names != dt.EXPECTED_CLASSES:
            problems.append(f"{cf} lists {names}, expected {dt.EXPECTED_CLASSES}")
        else:
            print(f"OK classes.txt: {cf}  -> 0 = crop, 1 = weed")

    samples, source_of = [], {}
    for pd in photo_dirs:
        ld = label_dirs.get(norm(pd.name))
        imgs = sorted(p for p in pd.iterdir() if p.is_file() and p.suffix.lower() in dt.IMAGE_EXTENSIONS)
        found = 0
        for img in imgs:
            lbl = (ld / f"{img.stem}.txt") if ld else None
            if lbl is None or not lbl.is_file():
                lbl = img.with_suffix(".txt")
            uid = f"{norm(pd.name)}__{img.stem}"
            s = dt.Sample(stem=uid, image_path=img, label_path=lbl if lbl.is_file() else None)
            dt._check_image(s)
            s.problems = [p for p in s.problems if not p.startswith("EXIF rotation")]  # LabelImg + YOLO both apply EXIF
            if s.label_path is None:
                s.problems.append("missing label file")
            else:
                dt._parse_label(s, num_classes=2)
                found += 1
            samples.append(s)
            source_of[uid] = pd.name
        print(f"  {pd.name}: {len(imgs)} photos, {found} label files"
              f"{'' if ld else '  (no matching sub-folder in labels-root; looked next to photos)'}")

    if not samples:
        print(f"\n[STOP] No JPG/PNG photos found in the sub-folders of {a.photos_root}")
        return 1
    bad = [s for s in samples if s.problems]
    for s in bad:
        problems.append(f"{source_of[s.stem]}/{s.image_path.name}: {'; '.join(s.problems)}")
    if problems:
        print("\n[STOP] Fix these first (nothing was copied):")
        for p in problems:
            print("  -", p)
        return 1

    # near-identical shots stay together
    print(f"Grouping near-identical shots ({len(samples)} photos) ...")
    gids = dt.group_duplicates(dt.duplicate_distance_matrix(samples), DUP_THRESHOLD)
    groups = collections.defaultdict(list)
    for s, g in zip(samples, gids):
        groups[g].append(s)

    rng = random.Random(a.seed)
    split_of = {}
    by_source = collections.defaultdict(list)
    for g, ss in groups.items():
        by_source[source_of[ss[0].stem]].append(g)
    for src in sorted(by_source):
        gl = sorted(by_source[src])
        rng.shuffle(gl)
        total = sum(len(groups[g]) for g in gl)
        filled = dict.fromkeys(SPLITS, 0)
        for g in gl:
            sp = min(SPLITS, key=lambda k: filled[k] / total - {"train": .70, "val": .15, "test": .15}[k])
            for s in groups[g]:
                split_of[s.stem] = sp
            filled[sp] += len(groups[g])

    out = a.output
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            print(f"[STOP] {out} already exists. Re-run with --overwrite to rebuild it.")
            return 1
        if not (out / MARKER).exists():
            print(f"[STOP] Refusing to delete {out}: not created by this script.")
            return 1
        shutil.rmtree(out)
    for k in ("images", "labels"):
        for sp in SPLITS:
            (out / k / sp).mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("generated - safe to delete and rebuild\n", encoding="utf-8")

    before = {s.image_path: dt.file_md5(s.image_path) for s in samples}
    stats = {sp: collections.Counter() for sp in SPLITS}
    rows = []
    for s in samples:
        sp = split_of[s.stem]
        dst_name = f"{s.stem}{s.image_path.suffix.lower()}"
        shutil.copy2(s.image_path, out / "images" / sp / dst_name)
        (out / "labels" / sp / f"{s.stem}.txt").write_text(dt.clipped_label_text(s), encoding="utf-8")
        stats[sp]["images"] += 1
        stats[sp][f"from: {source_of[s.stem]}"] += 1
        for b in s.boxes:
            stats[sp]["crop boxes" if b.class_id == 0 else "weed boxes"] += 1
        rows.append([s.image_path.name, source_of[s.stem], gids[samples.index(s)], sp, len(s.boxes)])

    (out / "data.yaml").write_text(
        "# Generated by weed_detection/prepare_tulsi_dataset.py\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: crop\n  1: weed\n", encoding="utf-8")
    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_file", "source_folder", "near_duplicate_group", "split", "boxes"])
        w.writerows(rows)
    unchanged = all(dt.file_md5(p) == h for p, h in before.items())
    report = {"photos": len(samples), "near_duplicate_groups": len(groups),
              "splits": {sp: dict(c) for sp, c in stats.items()}, "seed": a.seed,
              "originals_unchanged": unchanged, "class_mapping": {0: "crop", 1: "weed"}}
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n============ TULSI DATASET READY ============")
    for sp in SPLITS:
        print(f"{sp:5s}: {dict(stats[sp])}")
    print(f"Near-identical groups: {len(groups)} (from {len(samples)} photos)")
    print(f"Original photos unchanged: {'YES' if unchanged else 'NO - investigate'}")
    print(f"data.yaml: {(out / 'data.yaml').resolve()}")
    return 0 if unchanged else 2


if __name__ == "__main__":
    sys.exit(main())
