# Weed detector - test split results

Evaluated 2026-10-12T23:44:20 on 190 held-out images.
Model: `C:\Users\Hrishikesh Khanna\Desktop\Smart_irrigation_system_new-main\weed_detection\models\weed_detector.pt` (SHA-256 `ac2e0080cfb69a5b...`)

## A. Detection metrics (Ultralytics)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.8557 | 0.8205 | 0.8959 | 0.5792 |
| crop | 0.7554 | 0.785 | 0.8494 | 0.5611 |
| weed | 0.9559 | 0.856 | 0.9423 | 0.5973 |

## B. Image-level weed check (confidence >= 0.25)

| | Model says weed | Model says no weed |
|---|---|---|
| Image really has weed | 98 | 0 |
| Image has no weed | 0 | 92 |

Accuracy 1.0, weed recall 1.0, weed precision 1.0.

These numbers describe this dataset's test photos only (one crop species, single plants on bare soil). They do not show the model works on other crops, other weed species, or dense field scenes.
