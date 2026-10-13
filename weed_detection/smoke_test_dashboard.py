"""
Headless check that EVERY dashboard page still renders without an exception.

Uses Streamlit's built-in test runner (streamlit.testing.v1.AppTest), which runs dashboard.py
like a browser would, but without opening one. It visits all six pages:
Overview, Irrigation, Weed Detection, Weather, Recommendations, AI Models.

Run it BEFORE the Stage 3 change (baseline) and AFTER it. Both runs should pass.
Note: the real dashboard code runs, so Open-Meteo is contacted and the ESP32 port is tried,
exactly as when you launch the app. File uploads cannot be simulated here; upload inference
is covered by smoke_test_detector.py.

Usage (PowerShell, from the project root):
    python weed_detection\\smoke_test_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard.py"
PAGES = ["🏠 Overview", "💧 Irrigation", "🌿 Weed Detection", "🌦 Weather", "🧪 Recommendations", "🤖 AI Models"]


# Windows: when output is redirected to a file, avoid crashes on emoji / non-ASCII text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.run()
    failures = 0
    for page in PAGES:
        at.radio(key="navigation_radio").set_value(page).run()
        errors = [e.value for e in at.exception]
        titles = [t.value for t in at.title]
        status = "PASS" if not errors else "FAIL"
        failures += bool(errors)
        print(f"  {status}  {page:20s} title={titles[:1]}")
        for e in errors:
            print(f"        exception: {str(e)[:300]}")
        if page == "🌿 Weed Detection" and not errors:
            sliders = [s.label for s in at.slider]
            print(f"        weed page widgets: sliders={sliders}, "
                  f"messages={[m.value[:60] for m in at.success] + [m.value[:60] for m in at.error]}")
    print("RESULT:", "PASSED - all pages render" if not failures else f"FAILED on {failures} page(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
