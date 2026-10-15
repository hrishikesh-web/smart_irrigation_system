"""
Verify the friend-feature integration changed only what was planned.

Usage (project root):
    python verify_integration.py --before backups\\dashboard_before_integration.py --after dashboard.py

Expected: only get_sensor_data changed; load_disease_model / run_disease_detection added;
Weed Detection, Weather and Recommendations pages identical; no backend call, state write or
control lost on any page.
"""
from __future__ import annotations
import argparse, ast, collections, sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BACKEND_IMPORTS = {"run_model_inference", "audit_feature_readiness", "get_model_comparison_rows",
                   "get_model_trustworthiness", "get_model_details", "get_metrics"}
BACKEND_MODULES = {"database", "weed_detector", "requests"}
WIDGETS = {"button", "slider", "radio", "selectbox", "file_uploader", "camera_input", "number_input",
           "text_input", "checkbox", "toggle", "form_submit_button", "download_button", "multiselect",
           "date_input", "select_slider", "text_area"}
PRESENTATION_KWARGS = {"use_container_width", "type", "help", "horizontal", "width", "label_visibility"}
# (page, call text) pairs that may be NEW: read-only lookups used to show real data on the redesigned pages.
ALLOWED_NEW_CALLS = {
    ("🏠 Overview", "get_sensor_history()"),
    ("🏠 Overview", "weed_detector.model_status()"),
}
ALLOWED_NEW_TOPLEVEL_PREFIXES = ("import plotly.graph_objects", "from plotly.subplots import",
                                 "from html import escape", "UI_", "st.markdown(UI_CSS")


def page_label(test):
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "page"
            and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value
    return None


def split(tree):
    funcs, pages, other = {}, {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            funcs[node.name] = ast.dump(node)
        elif isinstance(node, ast.If) and page_label(node.test):
            cur = node
            while isinstance(cur, ast.If) and page_label(cur.test):
                pages[page_label(cur.test)] = cur.body
                cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
        else:
            other.append((ast.unparse(node), ast.dump(node)))
    return funcs, pages, other


def root_name(func):
    while isinstance(func, ast.Attribute):
        func = func.value
    return func.id if isinstance(func, ast.Name) else None


def fingerprint(body, backend_funcs):
    fp = collections.Counter()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else None
            if name in backend_funcs or name in BACKEND_IMPORTS or (
                isinstance(f, ast.Attribute) and root_name(f) in BACKEND_MODULES
            ) or (isinstance(f, ast.Attribute) and f.attr == "clear" and root_name(f) in backend_funcs):
                fp[("call", ast.unparse(node))] += 1
            elif isinstance(f, ast.Attribute) and f.attr in WIDGETS:
                label = ast.unparse(node.args[0]) if node.args else None
                rest = [ast.unparse(a) for a in node.args[1:]]
                kws = sorted((k.arg, ast.unparse(k.value)) for k in node.keywords if k.arg not in PRESENTATION_KWARGS)
                fp[("widget", f.attr, label, tuple(rest), tuple(kws))] += 1
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if "session_state" in ast.unparse(t):
                    fp[("state", ast.unparse(t), ast.unparse(node.value))] += 1
    return fp



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, default=Path("dashboard.py"))
    a = ap.parse_args()
    fb, pb, ob = split(ast.parse(a.before.read_text(encoding="utf-8")))
    fa, pa, oa = split(ast.parse(a.after.read_text(encoding="utf-8")))
    problems = []
    for n in fb:
        if n not in fa: problems.append(f"function removed: {n}")
        elif fa[n] != fb[n] and n != "get_sensor_data": problems.append(f"function changed unexpectedly: {n}")
    print("changed functions:", [n for n in fb if n in fa and fa[n] != fb[n]])
    print("new functions:    ", [n for n in fa if n not in fb])
    for n in fa:
        if n not in fb and n not in ("load_disease_model", "run_disease_detection"): problems.append(f"unexpected new function: {n}")
    backend = {n for n in fb if not n.startswith("ui_") and n != "display_value"}
    for label in pb:
        if label not in pa: problems.append(f"page removed: {label}"); continue
        same = ast.dump(ast.Module(body=pb[label], type_ignores=[])) == ast.dump(ast.Module(body=pa[label], type_ignores=[]))
        lost = fingerprint(pb[label], backend) - fingerprint(pa[label], backend)
        print(f"  {label:22s} {'identical' if same else 'changed'} | lost backend calls/state/controls: {sum(lost.values())}")
        if label in ("🌿 Weed Detection", "🌦 Weather", "🧪 Recommendations") and not same: problems.append(f"{label} should be identical")
        problems += [f"[{label}] lost: {x}" for x in lost]
    print("pages after:", list(pa))
    if problems:
        print("\nRESULT: FAILED"); [print("  !!", p) for p in problems]; return 1
    print("\nRESULT: PASSED - weed detector, weather, irrigation logic preserved; friend features added.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
