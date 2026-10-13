"""
Prove that the Stage 3 edit to dashboard.py changed ONLY the weed-detection parts.

It compares the backup and the new dashboard.py as parsed Python code (not as text), so
spacing does not matter. It checks that:
  - every function except run_weed_detection is identical (load_weed_model may be new),
  - every page branch except "Weed Detection" is identical: Overview, Irrigation, Weather,
    Recommendations, AI Models,
  - every other top-level statement is identical, apart from the expected weed import block
    and the two new session-state keys.

Usage (PowerShell, from the project root):
    python weed_detection\\verify_dashboard_patch.py --before backups\\dashboard_before_weed_stage3.py --after dashboard.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

WEED_PAGE = "🌿 Weed Detection"
ALLOWED_CHANGED_FUNCS = {"run_weed_detection"}
ALLOWED_NEW_FUNCS = {"load_weed_model"}
MUST_BE_IDENTICAL = ["get_weather_data", "get_rain_summary", "predict_irrigation", "get_sensor_data",
                     "get_thingspeak_data", "get_esp32_reader", "get_sensor_history", "get_water_history",
                     "get_treatment_recommendation", "display_value"]


def page_label(test: ast.expr) -> str | None:
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "page"
            and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value
    return None


def split(tree: ast.Module):
    funcs, pages, other = {}, {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            funcs[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.If) and page_label(node.test):
            cur = node
            while isinstance(cur, ast.If) and page_label(cur.test):
                pages[page_label(cur.test)] = ast.dump(ast.Module(body=cur.body, type_ignores=[]))
                cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
        else:
            other.append((ast.unparse(node).splitlines()[0][:80], ast.dump(node, include_attributes=False)))
    return funcs, pages, other


# Windows: when output is redirected to a file, avoid crashes on emoji / non-ASCII text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, default=Path("dashboard.py"))
    a = ap.parse_args()
    fb, pb, ob = split(ast.parse(a.before.read_text(encoding="utf-8")))
    fa, pa, oa = split(ast.parse(a.after.read_text(encoding="utf-8")))
    problems, notes = [], []

    for name in sorted(set(fb) | set(fa)):
        if name not in fa:
            problems.append(f"function removed: {name}")
        elif name not in fb:
            (notes if name in ALLOWED_NEW_FUNCS else problems).append(f"function added: {name}")
        elif fb[name] != fa[name]:
            (notes if name in ALLOWED_CHANGED_FUNCS else problems).append(f"function changed: {name}")
    for name in MUST_BE_IDENTICAL:
        if name in fb and fb.get(name) == fa.get(name):
            notes.append(f"identical: {name}")

    if list(pb) != list(pa):
        problems.append(f"page list/order changed: {list(pb)} -> {list(pa)}")
    for label in pb:
        if label in pa and pb[label] != pa[label]:
            (notes if label == WEED_PAGE else problems).append(f"page changed: {label}")
        elif label in pa:
            notes.append(f"page identical: {label}")

    before_dumps = [d for _, d in ob]
    after_dumps = [d for _, d in oa]
    for text, d in ob:
        if d not in after_dumps:
            (notes if text.startswith("DEFAULT_STATE") else problems).append(f"top-level statement changed/removed: {text}")
    for text, d in oa:
        if d not in before_dumps:
            expected = (text.startswith(("DEFAULT_STATE", "if str(PROJECT_ROOT) not in sys.path"))
                        or (text.startswith("try:") and "weed_detection" in d))
            (notes if expected else problems).append(f"top-level statement added/changed: {text}")

    print("Expected differences / confirmations:")
    for n in notes:
        print("  ok  ", n)
    if problems:
        print("\nUNEXPECTED CHANGES:")
        for p in problems:
            print("  !!  ", p)
        print("\nRESULT: FAILED - something outside the weed feature changed.")
        return 1
    print("\nRESULT: PASSED - only the weed-detection parts of dashboard.py changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
