"""
Shared helper functions for the NEW weed-detection pipeline.

This file is used by:
    check_dataset.py    (read-only inspection)
    prepare_dataset.py  (builds the train/val/test copy)

Nothing in this file writes to the original dataset. It only READS it.
It does not import or use any of the older weed-detection code in training/.
"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# The class names we expect, in the order we expect them.
# check_dataset.py verifies the real classes file matches this before anything else runs.
EXPECTED_CLASSES = ["crop", "weed"]

# Two images whose "visual fingerprints" differ by this many bits (out of 256) or fewer
# are treated as copies of the same original photo. Chosen from a calibration on this
# dataset: grouping is stable from 40 to 72, and unrelated images start merging at about 88.
DEFAULT_DUPLICATE_THRESHOLD = 56

# Boxes may poke this far outside the image (as a fraction of image size) and will
# simply be clipped. Anything further outside is reported as an error.
EDGE_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Box:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Sample:
    stem: str                      # file name without extension, e.g. agri_0_1009
    image_path: Path
    label_path: Path | None
    boxes: list[Box] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None

    @property
    def class_set(self) -> tuple[int, ...]:
        return tuple(sorted({b.class_id for b in self.boxes}))

    @property
    def dominant_class(self) -> int | None:
        """The class that covers the most box area in this image."""
        if not self.boxes:
            return None
        area = collections.Counter()
        for b in self.boxes:
            area[b.class_id] += b.area
        return max(area, key=lambda c: (area[c], -c))


# ---------------------------------------------------------------------------
# Locating files
# ---------------------------------------------------------------------------

def find_classes_file(source: Path) -> Path:
    for name in ("classes.txt", "classes"):
        p = source / name
        if p.is_file():
            return p
    hits = [p for p in source.rglob("classes*") if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"No 'classes' or 'classes.txt' file found under {source}")
    return hits[0]


def find_image_folder(source: Path) -> Path:
    """Return the folder that holds the most images (agri_data/data in this dataset)."""
    counts = collections.Counter()
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            counts[p.parent] += 1
    if not counts:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png images found under {source}")
    return counts.most_common(1)[0][0]


def read_classes(path: Path) -> list[str]:
    # utf-8-sig removes a hidden BOM; strip() removes Windows line endings (\r).
    text = path.read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Reading and validating samples
# ---------------------------------------------------------------------------

def load_samples(image_folder: Path, num_classes: int) -> tuple[list[Sample], list[str]]:
    """Read every image and its matching .txt label. Returns (samples, orphan_label_files)."""
    images = {}
    for p in sorted(image_folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem in images:
                raise ValueError(f"Two images share the name {p.stem}: {images[p.stem].name}, {p.name}")
            images[p.stem] = p
    label_stems = {p.stem for p in image_folder.glob("*.txt")}
    orphans = sorted(label_stems - set(images))

    samples = []
    for stem, img_path in images.items():
        lbl = img_path.with_suffix(".txt")
        s = Sample(stem=stem, image_path=img_path, label_path=lbl if lbl.is_file() else None)
        _check_image(s)
        if s.label_path is None:
            s.problems.append("missing label file")
        else:
            _parse_label(s, num_classes)
        samples.append(s)
    return samples, orphans


def _check_image(s: Sample) -> None:
    try:
        with Image.open(s.image_path) as im:
            im.verify()
        with Image.open(s.image_path) as im:
            s.width, s.height = im.size
            orientation = im.getexif().get(0x0112, 1)
            if orientation not in (1, None):
                s.problems.append(f"EXIF rotation tag = {orientation} (boxes may not line up)")
    except Exception as exc:  # noqa: BLE001 - we want to report any decoding failure
        s.problems.append(f"image cannot be opened: {exc}")


def _parse_label(s: Sample, num_classes: int) -> None:
    text = s.label_path.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        s.problems.append("empty label file (no boxes)")
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            s.problems.append(f"line {n}: expected 5 numbers, got {len(parts)}: '{line}'")
            continue
        try:
            cls_f, cx, cy, w, h = map(float, parts)
        except ValueError:
            s.problems.append(f"line {n}: not numeric: '{line}'")
            continue
        if cls_f != int(cls_f) or not (0 <= int(cls_f) < num_classes):
            s.problems.append(f"line {n}: class id {parts[0]} is not 0..{num_classes - 1}")
            continue
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
            s.problems.append(f"line {n}: coordinates outside 0..1: '{line}'")
            continue
        t = EDGE_TOLERANCE
        if cx - w / 2 < -t or cx + w / 2 > 1 + t or cy - h / 2 < -t or cy + h / 2 > 1 + t:
            s.problems.append(f"line {n}: box extends outside the image: '{line}'")
            continue
        s.boxes.append(Box(int(cls_f), cx, cy, w, h))


def clipped_label_text(s: Sample) -> str:
    """YOLO label text for a sample, with tiny edge overshoots clipped into the image."""
    out = []
    for b in s.boxes:
        x1, y1 = max(0.0, b.cx - b.w / 2), max(0.0, b.cy - b.h / 2)
        x2, y2 = min(1.0, b.cx + b.w / 2), min(1.0, b.cy + b.h / 2)
        out.append(f"{b.class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
# Each image gets a 256-bit "difference hash": shrink to 17x16 grey pixels and record
# whether each pixel is brighter than its right-hand neighbour. Similar-looking images
# get similar bit patterns. We also hash all 8 flipped/rotated versions, so a mirrored
# or rotated copy of a photo is still recognised as the same photo.

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dhash_bits(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def _eight_orientations(im: Image.Image) -> list[Image.Image]:
    out = []
    for k in range(4):
        r = im.rotate(90 * k, expand=True)
        out += [r, ImageOps.mirror(r)]
    return out


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def duplicate_distance_matrix(samples: list[Sample]) -> np.ndarray:
    """Pairwise distance = fewest differing bits over all 8 orientations of the second image."""
    base, variants = [], []
    for s in samples:
        with Image.open(s.image_path) as im:
            im = im.convert("RGB")
            base.append(_dhash_bits(im))
            variants.append(np.stack([_dhash_bits(v) for v in _eight_orientations(im)]))
    base = np.stack(base)          # (N, 32) bytes
    variants = np.stack(variants)  # (N, 8, 32) bytes
    n = len(samples)
    dist = np.empty((n, n), dtype=np.int16)
    for i in range(n):
        xor = np.bitwise_xor(variants, base[i][None, None, :])
        dist[i] = _POPCOUNT[xor].sum(axis=2).min(axis=1)
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 0)
    return dist


def group_duplicates(dist: np.ndarray, threshold: int) -> list[int]:
    """Union-find: images closer than `threshold` (directly or via a chain) share a group id."""
    n = len(dist)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(np.triu(dist <= threshold, 1)):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


# ---------------------------------------------------------------------------
# Contradictory-label groups
# ---------------------------------------------------------------------------

def analyse_groups(samples: list[Sample], group_ids: list[int], class_names: list[str]) -> list[dict]:
    """For each duplicate group, decide whether its copies agree on the main class."""
    members = collections.defaultdict(list)
    for s, g in zip(samples, group_ids):
        members[g].append(s)
    groups = []
    for g, ss in sorted(members.items()):
        votes = collections.Counter(s.dominant_class for s in ss if s.dominant_class is not None)
        ranked = votes.most_common()
        tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
        majority = None if (not ranked or tie) else ranked[0][0]
        groups.append({
            "group_id": g,
            "size": len(ss),
            "stems": [s.stem for s in ss],
            "votes": {class_names[c]: v for c, v in votes.items()},
            "contradictory": len(votes) > 1,
            "tie": tie,
            "majority_class": class_names[majority] if majority is not None else None,
            "majority_class_id": majority,
        })
    return groups
