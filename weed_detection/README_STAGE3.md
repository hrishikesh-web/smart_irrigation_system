# Weed Detection – Stage 3: Dashboard integration (PROPOSAL – not applied yet)

## Files

| File | Change |
|---|---|
| `dashboard.py` | **Edited in 4 places only** (see below). Backed up first. |
| `weed_detection/__init__.py` | New, 1 line. Lets the dashboard `import weed_detection.detector` without touching `sys.path` order |
| `weed_detection/detector.py` | New. Loads `models/weed_detector.pt`, checks images, runs detection. No Streamlit code. |
| `weed_detection/smoke_test_detector.py` | New. Upload-inference smoke test (JPEG, PNG, bad files, missing model) |
| `weed_detection/smoke_test_dashboard.py` | New. Opens every dashboard page headlessly and fails on any exception |
| `weed_detection/verify_dashboard_patch.py` | New. Proves only the weed parts of `dashboard.py` changed |

The 4 edits in `dashboard.py`:

1. **After the existing `inference` import block:** import `weed_detection.detector` inside
   `try/except`. If it fails, only the Weed page shows an error. Ultralytics is not imported here.
2. **`DEFAULT_STATE`:** two new keys, `weed_result` and `weed_result_key`. All existing keys are kept.
3. **The placeholder `run_weed_detection()`** is replaced by a real one, plus `load_weed_model()`
   wrapped in `@st.cache_resource`, so the model loads once, not on every rerun. The original
   return keys `weed_detected`, `weed_name` and `confidence` are kept for the other pages.
4. **The `elif page == "🌿 Weed Detection":` block:** adds a confidence slider, a **Detect Weeds**
   button, side-by-side original and annotated images, counts, a detection table and the
   "No weeds detected" message. The upload/camera choice and the "Detection Result" section are kept.

Not touched: page navigation, sidebar, sensor/ESP32 code, the Open-Meteo weather function,
irrigation prediction, the Recommendations / Weather / Irrigation / AI Models / Overview pages,
the old `YOLO_MODEL_PATH` constant (left unused) and `training/`.

## How to apply (only after review)

```powershell
cd "C:\Users\YOUR_NAME\Projects\Smart_irrigation_system_new-main"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 1. Is your dashboard.py exactly the version the proposal was made from?
(Get-FileHash dashboard.py -Algorithm MD5).Hash      # expected: C090F55490C516FF187C80E1258AE68C
#    If it is different, STOP and tell me - the proposal must be re-based on your current file.

# 2. Back up dashboard.py
New-Item -ItemType Directory -Force backups | Out-Null
Copy-Item dashboard.py backups\dashboard_before_weed_stage3.py
(Get-FileHash backups\dashboard_before_weed_stage3.py -Algorithm MD5).Hash   # same hash as step 1

# 3. Unzip the new weed_detection files (they do not change the dashboard by themselves)
Expand-Archive -Path "$HOME\Downloads\weed_stage3_review.zip" -DestinationPath stage3_review -Force
Copy-Item stage3_review\weed_detection\* weed_detection\ -Force

# 4. BASELINE - record how things behave BEFORE the dashboard edit
python -m unittest test_weather_integration -v *> backups\weather_tests_before.txt
python -m unittest test_ml_integration -v      *> backups\ml_tests_before.txt
python weed_detection\smoke_test_dashboard.py  *> backups\pages_before.txt
Get-Content backups\pages_before.txt

# 5. Apply the dashboard edit
Copy-Item stage3_review\dashboard_PROPOSED.py dashboard.py -Force

# 6. AFTER - same tests, plus the two new checks
python weed_detection\verify_dashboard_patch.py --before backups\dashboard_before_weed_stage3.py --after dashboard.py
python -m unittest test_weather_integration -v *> backups\weather_tests_after.txt
python -m unittest test_ml_integration -v      *> backups\ml_tests_after.txt
python weed_detection\smoke_test_dashboard.py  *> backups\pages_after.txt
Get-Content backups\pages_after.txt
python weed_detection\smoke_test_detector.py

# 7. Compare before/after test summaries (the last lines should match: same "Ran N tests ... OK")
Get-Content backups\weather_tests_before.txt -Tail 3; Get-Content backups\weather_tests_after.txt -Tail 3
Get-Content backups\ml_tests_before.txt -Tail 3;      Get-Content backups\ml_tests_after.txt -Tail 3
```

**Undo at any time:** `Copy-Item backups\dashboard_before_weed_stage3.py dashboard.py -Force`

## Launch the dashboard

```powershell
streamlit run dashboard.py
```

It opens `http://localhost:8501` in your browser (open it yourself if it doesn't). Stop it with **Ctrl + C**.

## Manual check in the browser

1. **🌦 Weather:** Vellore temperature, humidity, wind and forecast still appear (needs internet).
2. **💧 Irrigation:** the irrigation decision and model output appear as before.
3. **🤖 AI Models, 🏠 Overview, 🧪 Recommendations:** they open with no red error box.
4. **🌿 Weed Detection:**
   - you see "Model file found";
   - upload a test image from `weed_detection\dataset\images\test` and click **Detect Weeds**;
   - boxes appear with "crop"/"weed" labels and scores;
   - drag the slider to 0.95 and click again; fewer boxes, possibly "No weeds detected";
   - the first detection takes a few seconds (the model loads once), and later ones are fast.
5. **🧪 Recommendations:** "Detected Weed" now shows the last result.
