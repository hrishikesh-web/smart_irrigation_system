# Weed Detection – Stage 2: Training and test evaluation

Stage 2 adds three files to `weed_detection/` and changes nothing else in the project.
`dashboard.py`, the irrigation models and the Open-Meteo weather code are untouched.

| New file | What it does |
|---|---|
| `weed_settings.py` | Every path and default setting in one place (also used by the dashboard in Stage 3) |
| `train.py` | Checks everything, downloads the official checkpoint, fine-tunes it, saves the best model |
| `evaluate.py` | Measures the trained model on the 190 **held-out test** images |

## The idea in plain words

- **Pretrained checkpoint.** We don't start from a blank model. We start from **YOLO11n**,
  a small ("n" = nano) detector that Ultralytics has already trained on the large COCO photo
  set. It already knows edges, leaves, textures and shapes. We only teach it our two classes.
  This is called *fine-tuning*.
- **Where it comes from.** `train.py` downloads it from Ultralytics' official GitHub release:
  `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
  and checks its SHA-256 fingerprint
  (`0ebbc80d…6417644ee1`). If the fingerprint is wrong, training stops.
  The old `training/yolov8n.pt` and any other repository weights are never used.
- **Epoch.** One pass over all 884 training images. After each epoch the model is scored on
  the 190 *validation* images, and the best-scoring version is saved as `best.pt`.
- **Early stopping (patience 25).** If the validation score hasn't improved for 25 epochs,
  training stops early. Waiting longer would just waste time.
- **Test split.** The 190 *test* images are never used during training or to pick `best.pt`.
  `evaluate.py` uses them once at the end. Those are the honest numbers for your report.

## Settings (chosen for an 8 GB RTX 4060)

| Setting | Value | Why |
|---|---|---|
| Starting model | `yolo11n` (official) | Smallest, fast, well-tested; ~2.6 M parameters |
| Image size | 512 | The dataset images are 512 × 512 |
| Batch size | 16 | A nano model at 512 px uses well under 8 GB |
| Epochs | 80 (max) | Usually enough for a small dataset; early stopping may end sooner |
| Patience | 25 | Stop if no improvement for 25 epochs |
| Workers | 4 | Image-loading processes; use 2 or 0 if Windows has trouble |
| Seed | 42, deterministic on | Repeatable results (may still differ slightly on another GPU) |
| Augmentation | Ultralytics defaults | Flips, colour changes, mosaic. The dataset already contains rotated copies. |

Optional alternative: `--model yolo26n`, Ultralytics' newer nano model, also official and
fingerprint-checked. Stick with the default unless you have time to compare both.

## Exact PowerShell commands (run from the project root)

```powershell
# 0. Open PowerShell in the project folder and activate the environment
cd "C:\Users\YOUR_NAME\Projects\Smart_irrigation_system_new-main"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 1. Dry run: checks the dataset, GPU and Ultralytics, downloads + verifies yolo11n.pt.
#    Trains nothing. Every line should look fine and it should end with "No training was started."
python weed_detection\train.py --check-only

# 2. Full training (plug in the charger, and stop Windows from sleeping while it runs)
python weed_detection\train.py

#    Change settings if you need to:
#    python weed_detection\train.py --epochs 50
#    python weed_detection\train.py --batch 8          # if you see "CUDA out of memory"
#    python weed_detection\train.py --workers 0        # if you see DataLoader / worker errors

# 3. Evaluate on the held-out TEST images
python weed_detection\evaluate.py

# 4. Read the results
Get-Content weed_detection\reports\test_metrics.md
Start-Process (Get-ChildItem weed_detection\runs -Directory -Filter "train_*" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
```

While it trains, the console prints one line per epoch. **box_loss** and **cls_loss** should
generally go down, and **mAP50** on validation should generally go up. Ultralytics
also runs a quick mixed-precision self-test that downloads `yolo26n.pt` into its own
settings folder. That file is only used for the self-test, not for our training.

How long it takes depends on your laptop. For about 900 small images on an RTX 4060 it
should be minutes per handful of epochs, not hours. The console shows the real speed.

## Where everything is saved

```
weed_detection/
├── pretrained/yolo11n.pt               official starting checkpoint (verified download)
├── runs/
│   ├── train_yolo11n_512_<date-time>/  one folder per training run
│   │   ├── weights/best.pt             best epoch on VALIDATION  <- the trained model
│   │   ├── weights/last.pt             final epoch (used for nothing else)
│   │   ├── results.csv / results.png   loss and validation metrics per epoch
│   │   ├── confusion_matrix.png, PR curves, val_batch*_pred.jpg
│   │   ├── args.yaml                   every setting Ultralytics used
│   │   └── weed_detector_info.json     how this model was made (checkpoint, settings, GPU)
│   └── eval_test_<date-time>/          plots + example_predictions/ from evaluate.py
├── models/
│   ├── weed_detector.pt                copy of best.pt - what the dashboard will load in Stage 3
│   ├── weed_detector_info.json
│   └── archive/                        older models are moved here, never overwritten
└── reports/
    ├── test_metrics.json               all TEST numbers
    └── test_metrics.md                 the same as a readable table for your report
```

## What evaluate.py reports

- **A. Detection metrics** (standard Ultralytics validation on the test split):
  - *Precision*: of the boxes the model drew, the share that were correct.
  - *Recall*: of the real plants, the share the model found.
  - *mAP50*: overall detection quality when a box only needs 50% overlap with the true box.
    This is the most common headline number.
  - *mAP50-95*: a stricter average that also rewards tight boxes.
  - All of these are given overall and separately for **crop** and **weed**.
- **B. Image-level check**: for each test image, does the model report at least one weed at
  confidence ≥ 0.25 (the dashboard's default)? It's compared with the true labels in a 2 × 2 table.
  This is closest to what the "Weed Detection" page will show.

Only report numbers that `evaluate.py` actually printed, and say they come from 190 test
images (41 independent photo scenes) of one crop species on bare soil.

## Backup: train on Google Colab

Use this only if local training fails.

1. On Windows, after Stage 1, zip the folder (it includes the prepared dataset):
   ```powershell
   Compress-Archive -Path weed_detection -DestinationPath weed_detection_colab.zip -Force
   ```
2. Open https://colab.research.google.com → **New notebook** →
   **Runtime ▸ Change runtime type ▸ T4 GPU** → Save.
3. Run these cells one at a time:
   ```python
   !pip install -q ultralytics==8.4.175
   import torch; print(torch.__version__, torch.cuda.is_available())   # must print True
   ```
   ```python
   from google.colab import files
   files.upload()                      # choose weed_detection_colab.zip
   !mkdir -p /content/project && unzip -q -o weed_detection_colab.zip -d /content/project
   %cd /content/project
   ```
   ```python
   !python weed_detection/train.py --check-only
   !python weed_detection/train.py
   !python weed_detection/evaluate.py
   ```
   ```python
   !zip -q -r trained_model.zip weed_detection/models weed_detection/reports
   files.download("trained_model.zip")
   ```
4. Back on Windows, unpack it into the project root (it fills `weed_detection\models` and `reports`):
   ```powershell
   Expand-Archive -Path "$HOME\Downloads\trained_model.zip" -DestinationPath . -Force
   ```

`train.py` always uses the folder `data.yaml` is actually in, so the Windows path written in
Stage 1 doesn't break on Colab.

## Troubleshooting

| Message | Fix |
|---|---|
| `[STOP] PyTorch cannot see a CUDA GPU` | Follow the printed steps (`nvidia-smi`, reinstall the CUDA build of torch) |
| `CUDA out of memory` | `--batch 8` (or close other GPU apps) |
| `DataLoader worker ... exited unexpectedly` / `BrokenPipeError` | `--workers 0` |
| `Could not download the official checkpoint` | Download the printed URL in a browser and save it to the printed path |
| Training stopped (laptop slept / closed) | Run `python weed_detection\train.py` again; it starts a fresh run folder |
