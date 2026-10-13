"""
STEP 2 of the weed-detection pipeline: PREPARE a clean train / val / test COPY.

What it does
    1. Runs every check from check_dataset.py (stops if anything is fatally wrong).
    2. Removes images that have broken labels (none in the current dataset).
    3. Resolves "contradictory" duplicate groups (copies of one photo with different labels):
         --conflicts majority  (default) keep the copies that agree with the majority label,
                               drop the disagreeing copies; drop the whole group on a tie.
         --conflicts drop      drop every image in every contradictory group.
    4. Splits by DUPLICATE GROUP, not by single image, so all copies of one photo land in
       the same split (prevents the test set containing near-copies of training images).
       Default split: 70% train, 15% val, 15% test, balanced per class, fixed random seed.
    5. COPIES (never moves) images + labels into   weed_detection/dataset/
           images/train  images/val  images/test
           labels/train  labels/val  labels/test
       and writes data.yaml, which the YOLO trainer reads in Stage 2.
    6. Re-checks the fingerprints of the original files to prove nothing was changed.

Usage (PowerShell, from the project folder):
    python weed_detection\\prepare_dataset.py --source "C:\\path\\to\\weed_dataset"
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

import numpy as np

import dataset_tools as dt
from check_dataset import DEFAULT_REPORT_DIR, analyse_dataset, summarise, write_reports

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dataset"
MARKER = ".created_by_prepare_dataset"
SPLITS = ("train", "val", "test")


def choose_images(a: dict, policy: str) -> tuple[list[dt.Sample], dict[str, str]]:
    """Return (kept samples, {stem: reason} for excluded samples)."""
    excluded = {}
    for s in a["samples"]:
        if s.problems:
            excluded[s.stem] = "label/image problem: " + "; ".join(s.problems)
    by_stem = {s.stem: s for s in a["samples"]}
    for g in a["groups"]:
        if not g["contradictory"]:
            continue
        for stem in g["stems"]:
            if stem in excluded:
                continue
            if policy == "drop" or g["tie"]:
                excluded[stem] = f"contradictory group {g['group_id']} dropped ({'tie' if g['tie'] else 'policy=drop'})"
            elif by_stem[stem].dominant_class != g["majority_class_id"]:
                excluded[stem] = (f"contradictory group {g['group_id']}: label disagrees with majority "
                                  f"'{g['majority_class']}'")
    kept = [s for s in a["samples"] if s.stem not in excluded]
    return kept, excluded


def split_groups(kept: list[dt.Sample], group_ids: dict[str, int], ratios: tuple[float, float, float],
                 seed: int) -> dict[int, str]:
    """Assign every duplicate group to one split, balancing image counts within each class."""
    groups = collections.defaultdict(list)
    for s in kept:
        groups[group_ids[s.stem]].append(s)
    by_class = collections.defaultdict(list)
    for gid, ss in groups.items():
        main = collections.Counter(s.dominant_class for s in ss).most_common(1)[0][0]
        by_class[main].append(gid)

    rng = random.Random(seed)
    assignment = {}
    for cls in sorted(by_class):
        gids = sorted(by_class[cls])
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        filled = dict.fromkeys(SPLITS, 0)
        for gid in gids:
            # put this group in whichever split is furthest below its target share
            split = min(SPLITS, key=lambda sp: filled[sp] / total - ratios[SPLITS.index(sp)])
            assignment[gid] = split
            filled[split] += len(groups[gid])
    return assignment


def safe_prepare_output(out: Path, source: Path, overwrite: bool) -> None:
    out_r, src_r = out.resolve(), source.resolve()
    if out_r == src_r or src_r in out_r.parents or out_r in src_r.parents:
        raise SystemExit(f"Refusing: output folder {out_r} overlaps the original dataset {src_r}.")
    if out_r.exists() and any(out_r.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output folder already exists: {out_r}\n"
                             "Re-run with --overwrite to rebuild it (only folders made by this script are deleted).")
        if not (out_r / MARKER).exists():
            raise SystemExit(f"Refusing to delete {out_r}: it was not created by prepare_dataset.py.")
        shutil.rmtree(out_r)
    for kind in ("images", "labels"):
        for sp in SPLITS:
            (out_r / kind / sp).mkdir(parents=True, exist_ok=True)
    (out_r / MARKER).write_text("This folder is generated. Safe to delete and rebuild.\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a leak-free train/val/test copy of the crop/weed dataset.")
    ap.add_argument("--source", required=True, type=Path, help="Unzipped weed_dataset folder (left untouched).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to build the copy.")
    ap.add_argument("--conflicts", choices=["majority", "drop"], default="majority")
    ap.add_argument("--threshold", type=int, default=dt.DEFAULT_DUPLICATE_THRESHOLD)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="Rebuild an existing output folder.")
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1) > 1e-6:
        raise SystemExit("--train + --val + --test must add up to 1.0")

    a = analyse_dataset(args.source, args.threshold)
    summary = summarise(a)
    write_reports(a, summary, args.report_dir)
    if summary["fatal_errors"]:
        print("FATAL problems found - nothing was prepared:")
        for e in summary["fatal_errors"]:
            print("  -", e)
        return 1

    names = a["class_names"]
    gid_of = {s.stem: g for s, g in zip(a["samples"], a["group_ids"])}
    kept, excluded = choose_images(a, args.conflicts)
    assignment = split_groups(kept, gid_of, ratios, args.seed)
    split_of = {s.stem: assignment[gid_of[s.stem]] for s in kept}

    # Leakage check: closest pair of images that ended up in DIFFERENT splits.
    index = {s.stem: i for i, s in enumerate(a["samples"])}
    idx = np.array([index[s.stem] for s in kept])
    lab = np.array([SPLITS.index(split_of[s.stem]) for s in kept])
    sub = a["distance"][np.ix_(idx, idx)]
    cross = sub[lab[:, None] != lab[None, :]]
    closest_cross = int(cross.min()) if cross.size else None
    groups_in_many_splits = sum(len({split_of[s] for s in g["stems"] if s in split_of}) > 1 for g in a["groups"])

    out = args.output
    safe_prepare_output(out, a["source"], args.overwrite)
    print(f"Copying {len(kept)} images into {out.resolve()} ...")
    for s in kept:
        sp = split_of[s.stem]
        shutil.copy2(s.image_path, out / "images" / sp / s.image_path.name)
        (out / "labels" / sp / f"{s.stem}.txt").write_text(dt.clipped_label_text(s), encoding="utf-8", newline="\n")

    yaml_text = (
        "# Generated by weed_detection/prepare_dataset.py - read by YOLO training.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8", newline="\n")

    # Prove the original dataset is unchanged.
    after = {str(p.relative_to(a["source"])).replace("\\", "/"): dt.file_md5(p)
             for p in sorted(a["source"].rglob("*")) if p.is_file()}
    original_unchanged = after == a["fingerprint"]

    # Per-split statistics.
    stats = {sp: {"images": 0, "groups": set(), "boxes": collections.Counter(), "main_class": collections.Counter()}
             for sp in SPLITS}
    for s in kept:
        st = stats[split_of[s.stem]]
        st["images"] += 1
        st["groups"].add(gid_of[s.stem])
        st["main_class"][names[s.dominant_class]] += 1
        for b in s.boxes:
            st["boxes"][names[b.class_id]] += 1
    split_stats = {sp: {"images": v["images"], "duplicate_groups": len(v["groups"]),
                        "images_by_main_class": dict(v["main_class"]), "boxes_by_class": dict(v["boxes"])}
                   for sp, v in stats.items()}

    excl_reasons = collections.Counter(r.split(":")[0] if "problem" in r else
                                       ("tie" if "tie" in r else "disagrees with majority" if "majority" in r
                                        else "policy=drop") for r in excluded.values())
    report = {
        "source": str(a["source"]), "output": str(out.resolve()),
        "conflict_policy": args.conflicts, "ratios": dict(zip(SPLITS, ratios)), "seed": args.seed,
        "duplicate_threshold": args.threshold,
        "images_total": len(a["samples"]), "images_kept": len(kept), "images_excluded": len(excluded),
        "excluded_by_reason": dict(excl_reasons),
        "splits": split_stats,
        "leakage_check": {
            "groups_split_across_sets": groups_in_many_splits,
            "closest_cross_split_distance": closest_cross,
            "threshold": args.threshold,
            "passed": groups_in_many_splits == 0 and (closest_cross is None or closest_cross > args.threshold),
        },
        "original_dataset_unchanged": original_unchanged,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(args.report_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "duplicate_group", "main_class", "split", "status", "reason"])
        for s in a["samples"]:
            main_cls = names[s.dominant_class] if s.dominant_class is not None else ""
            if s.stem in excluded:
                w.writerow([s.stem, gid_of[s.stem], main_cls, "", "excluded", excluded[s.stem]])
            else:
                w.writerow([s.stem, gid_of[s.stem], main_cls, split_of[s.stem], "kept", ""])

    print("\n================ SPLIT SUMMARY ================")
    print(f"Images kept {len(kept)} / {len(a['samples'])}   excluded {len(excluded)}  {dict(excl_reasons)}")
    for sp in SPLITS:
        v = split_stats[sp]
        print(f"{sp:5s}: {v['images']:4d} images, {v['duplicate_groups']:3d} groups, "
              f"main class {v['images_by_main_class']}, boxes {v['boxes_by_class']}")
    lc = report["leakage_check"]
    print(f"Leakage check: groups across splits = {lc['groups_split_across_sets']}, closest cross-split "
          f"distance = {lc['closest_cross_split_distance']} (must be > {args.threshold}) -> "
          f"{'PASSED' if lc['passed'] else 'FAILED'}")
    print(f"Original dataset unchanged: {'YES' if original_unchanged else 'NO - investigate!'}")
    print(f"data.yaml : {(out / 'data.yaml').resolve()}")
    print(f"Reports   : {args.report_dir.resolve()}")
    return 0 if (lc["passed"] and original_unchanged) else 2


if __name__ == "__main__":
    sys.exit(main())
