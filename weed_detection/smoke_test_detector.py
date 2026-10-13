"""
Smoke test for upload inference - runs the SAME detector code the dashboard uses, without Streamlit.

Checks
  1. the trained model loads and has classes {0: crop, 1: weed}
  2. JPEG bytes from a held-out TEST image (one crop-only, one with weed) give an annotated image
  3. the same image re-encoded as PNG (with transparency) also works
  4. garbage bytes, an empty file and a GIF are rejected with a clear WeedImageError
  5. a missing model path gives a clear WeedModelError
  6. optional: your own photos via --extra
Annotated outputs are saved to weed_detection/reports/smoke_test/ so you can look at them.

It reports what the model detected but does not fail on accuracy. Accuracy comes from evaluate.py.

Usage (PowerShell, from the project root):
    python weed_detection\\smoke_test_detector.py
    python weed_detection\\smoke_test_detector.py --extra "C:\\path\\photo1.jpg" "C:\\path\\photo2.png"
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, so the package import works
from weed_detection import detector, weed_settings as cfg  # noqa: E402

OUT = cfg.REPORTS_DIR / "smoke_test"
failures = []


def check(ok: bool, label: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        failures.append(label)


def pick_test_images() -> dict[str, Path]:
    img_dir, lbl_dir = cfg.WEED_DIR / "dataset" / "images" / "test", cfg.WEED_DIR / "dataset" / "labels" / "test"
    picks = {}
    for p in sorted(img_dir.iterdir()):
        classes = {ln.split()[0] for ln in (lbl_dir / f"{p.stem}.txt").read_text().splitlines() if ln.strip()}
        if classes == {"0"} and "crop_only" not in picks:
            picks["crop_only"] = p
        if "1" in classes and "with_weed" not in picks:
            picks["with_weed"] = p
    return picks


def describe(name: str, res: dict) -> None:
    dets = ", ".join(f"{d['class']} {d['confidence']:.2f}" for d in res["detections"]) or "nothing"
    print(f"        {name}: weeds={res['weed_count']} crops={res['crop_count']} -> {dets}")
    OUT.mkdir(parents=True, exist_ok=True)
    res["annotated"].save(OUT / f"{name}_annotated.jpg", quality=90)


# Windows: when output is redirected to a file, avoid crashes on emoji / non-ASCII text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=cfg.DEFAULT_CONFIDENCE)
    ap.add_argument("--extra", nargs="*", type=Path, default=[], help="your own JPG/PNG photos to try")
    args = ap.parse_args()

    print("1. Model loading")
    t = time.time()
    try:
        model = detector.load_model(cfg.FINAL_MODEL)
        check(True, f"loaded {cfg.FINAL_MODEL} in {time.time() - t:.1f}s, classes {model.names}")
    except detector.WeedModelError as exc:
        check(False, f"model load: {exc}")
        print("\nCannot continue without the model.")
        return 1

    print("2. JPEG inference on held-out test images")
    picks = pick_test_images()
    check(set(picks) == {"crop_only", "with_weed"}, "found a crop-only and a weed test image")
    for name, path in picks.items():
        try:
            res = detector.detect(model, path.read_bytes(), conf=args.conf)
            check(res["annotated"].size == Image.open(path).size, f"{name} ({path.name}) annotated image returned")
            describe(name, res)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name}: {exc}")

    print("3. PNG with transparency")
    rgba = Image.open(next(iter(picks.values()))).convert("RGBA")
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    try:
        res = detector.detect(model, buf.getvalue(), conf=args.conf)
        check(True, "PNG (RGBA) accepted")
        describe("png_rgba", res)
    except Exception as exc:  # noqa: BLE001
        check(False, f"PNG: {exc}")

    print("4. Bad inputs are rejected cleanly")
    gif = io.BytesIO()
    Image.new("RGB", (20, 20)).save(gif, format="GIF")
    for label, data in [("garbage bytes", b"this is not an image"), ("empty file", b""), ("GIF file", gif.getvalue())]:
        try:
            detector.detect(model, data, conf=args.conf)
            check(False, f"{label} should have been rejected")
        except detector.WeedImageError as exc:
            check(True, f"{label} rejected: {exc}")

    print("5. Missing model file")
    try:
        detector.load_model(cfg.MODELS_DIR / "does_not_exist.pt")
        check(False, "missing model should raise")
    except detector.WeedModelError as exc:
        check(True, f"missing model reported: {str(exc)[:70]}...")

    if args.extra:
        print("6. Your own photos (no right/wrong check - look at the saved images)")
        for p in args.extra:
            try:
                describe(f"extra_{p.stem}", detector.detect(model, p.read_bytes(), conf=args.conf))
            except Exception as exc:  # noqa: BLE001
                print(f"        {p}: {exc}")

    print(f"\nAnnotated images saved in {OUT}")
    print("RESULT:", "PASSED" if not failures else f"FAILED ({len(failures)} problem(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
