# Weed Detection – Stage 1: Dataset check and preparation

This folder is a **new, separate** weed-detection pipeline. It does not use any of the
older weed files in `training/` (`detect_weed_demo.py`, `yolov8n.pt`) and it does not
touch `dashboard.py`, the irrigation models, or the Open-Meteo weather code.

## What is in this folder

| File | What it is | Changes anything? |
|---|---|---|
| `dataset_tools.py` | Shared helper functions (reading labels, finding duplicates) | No |
| `check_dataset.py` | **Read-only** inspection of the original dataset, writes reports | Only writes to `weed_detection/reports/` |
| `prepare_dataset.py` | Builds a train/val/test **copy** in `weed_detection/dataset/` | Only writes to `dataset/` and `reports/` |
| `.gitignore` | Stops the generated dataset copy being committed to Git | – |
| `README_STAGE1.md` | This guide | – |

The original dataset is only ever **read**. `prepare_dataset.py` fingerprints every
original file before and after, and prints `Original dataset unchanged: YES`.

## Key ideas (beginner version)

- **YOLO label format.** Each image `agri_0_123.jpeg` has a text file `agri_0_123.txt`.
  Each line is one box: `class  x_centre  y_centre  width  height`, with all numbers
  between 0 and 1 (fractions of the image size). Class `0` = crop and `1` = weed,
  taken from `classes.txt`.
- **Why duplicates matter.** This dataset is about 290 original photos, each saved
  4–17 times flipped, rotated, cropped or brightened. If one copy lands in *training* and
  another in *testing*, the model is tested on photos it has effectively already seen, and
  the score looks better than it really is. So the script groups copies together and
  keeps each group inside one split.
- **Contradictory groups.** In 35 groups, one copy of the same plant has the opposite label
  from the others. We keep the label most copies agree on and drop the odd one out (34 images).
  One group is a 1-vs-1 tie, so both copies are dropped (2 images).
- **Splits.** *train* is what the model learns from, *val* is used during training to pick
  the best checkpoint, and *test* is never seen until the final evaluation.

## Exact Windows PowerShell commands

Replace the two paths in step 1 with your real folders. Keep the double quotes,
because paths with spaces need them.

```powershell
# 1. Go to your project folder and remember where the unzipped dataset is
cd "C:\Users\YOUR_NAME\Projects\Smart_irrigation_system_new-main"
$DATA = "C:\Users\YOUR_NAME\Projects\weed_dataset"

# (only if the dataset is still a zip) unzip it; the folder must then contain classes.txt and agri_data
Expand-Archive -Path "$HOME\Downloads\weed_dataset.zip" -DestinationPath $DATA
Get-ChildItem $DATA          # you should see: agri_data  and  classes.txt

# 2. Activate the project's Python environment
#    If you get "running scripts is disabled", run the Set-ExecutionPolicy line first.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
#    No .venv folder yet? Create one first:  python -m venv .venv   then run the line above again.

# 3. Stage 1 only needs Pillow and NumPy (Streamlit already installs both). Check:
python -c "import PIL, numpy; print('Pillow', PIL.__version__, '| NumPy', numpy.__version__)"
#    If that errors:  python -m pip install pillow numpy

# 4. Read-only check of the original dataset (takes about 30 seconds)
python weed_detection\check_dataset.py --source "$DATA"

# 5. Look at the pictures it made
Start-Process weed_detection\reports\class_samples.jpg        # blue = crop, red = weed
Start-Process weed_detection\reports\contradictory_groups.jpg  # keep / DROP decisions

# 6. Build the train/val/test copy
python weed_detection\prepare_dataset.py --source "$DATA"
#    To rebuild later:  add  --overwrite
#    To drop every contradictory group instead of majority vote:  add  --conflicts drop

# 7. Confirm the result
Get-Content weed_detection\dataset\data.yaml
(Get-ChildItem weed_detection\dataset\images\train).Count   # expect 884
(Get-ChildItem weed_detection\dataset\images\val).Count     # expect 190
(Get-ChildItem weed_detection\dataset\images\test).Count    # expect 190
```

The split uses a fixed random seed (42), so running it on your laptop should give the
same 884 / 190 / 190 split shown below.

## Results on the uploaded dataset

| Check | Result |
|---|---|
| Classes file | `crop` = 0, `weed` = 1 (confirmed by eye on sample images) |
| Images / labels | 1,300 / 1,300, all matched, no orphans |
| Broken images or label lines | 0 |
| Image size | all 512 × 512 RGB |
| Boxes | 1,212 crop, 860 weed (296 are tiny, under 1% of the image) |
| Exact byte-for-byte duplicates | 0 |
| Near-duplicate groups (incl. flips/rotations) | 289 groups, largest has 17 copies, no image without copies |
| Contradictory groups | 35 (221 images): 34 resolved by majority, 1 tie dropped |
| Images kept | 1,264 of 1,300 (36 excluded) |
| train / val / test | 884 / 190 / 190 images (204 / 43 / 41 photo groups) |
| Leakage check | 0 groups split across sets; closest cross-split pair distance 70 (> 56): **PASSED** |
| Original dataset | unchanged (fingerprint check): **YES** |

## Outputs

```
weed_detection/
├── dataset/                    <- generated copy (safe to delete and rebuild)
│   ├── data.yaml               <- tells YOLO where the images are and the class names
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── reports/
    ├── dataset_check.json      <- summary numbers
    ├── duplicate_groups.csv    <- which images are copies of each other
    ├── label_problems.csv      <- bad label lines (empty for this dataset)
    ├── source_fingerprint.csv  <- md5 of every original file
    ├── split_manifest.csv      <- every image: group, split, kept/excluded and why
    ├── split_report.json       <- per-split counts and the leakage check
    ├── class_samples.jpg
    └── contradictory_groups.jpg
```

## Limitations to state honestly in the report

- "crop" in this dataset is essentially **one crop species**, mostly photographed on
  cracked dry soil. The model will not know other crops.
- Only about 290 truly different photos exist, so the test set holds about 41 independent
  scenes. Treat metrics as a rough estimate.
- The dataset shows single plants on bare soil. Dense roadside vegetation (like your
  college photos) looks quite different, so expect weaker results on those.
