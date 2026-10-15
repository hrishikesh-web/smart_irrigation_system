
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "combined_data.csv"
MODEL_PATH = ROOT / "models" / "future_dryness_model.joblib"

FORECAST_MINUTES = 30
LOW_MOISTURE_PERCENTILE = 0.20

# Load and clean the dataset
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").drop_duplicates("timestamp")
df = df.reset_index(drop=True)

features = [
    "soil_moisture_1",
    "soil_moisture_2",
    "soil_temperature",
    "rainfall",
]

# Find a future moisture reading at or after 30 minutes ahead
future = df[["timestamp", "soil_moisture_1"]].copy()
future["target_time"] = (
    future["timestamp"] - pd.Timedelta(minutes=FORECAST_MINUTES)
)
future = future.rename(
    columns={"soil_moisture_1": "future_moisture"}
)

df = pd.merge_asof(
    df.sort_values("timestamp"),
    future[["target_time", "future_moisture"]].sort_values("target_time"),
    left_on="timestamp",
    right_on="target_time",
    direction="forward",
    tolerance=pd.Timedelta(minutes=5),
)

df = df.dropna(subset=["future_moisture"]).reset_index(drop=True)

# Chronological split: earlier observations for training,
# later observations for testing.
split = int(len(df) * 0.80)
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

# Define "low" from the training period only
threshold = train_df["future_moisture"].quantile(
    LOW_MOISTURE_PERCENTILE
)

y_train = (train_df["future_moisture"] <= threshold).astype(int)
y_test = (test_df["future_moisture"] <= threshold).astype(int)

X_train = train_df[features]
X_test = test_df[features]

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "The target has only one class in train or test. "
        "Inspect the data before proceeding."
    )

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Baseline: predict low moisture using the current reading
baseline_predictions = (
    X_test["soil_moisture_1"].to_numpy() <= threshold
).astype(int)

print("\n--- ML vs simple current-moisture rule ---")
print(
    "ML accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)
print(
    "Simple-rule accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)
print(
    "ML balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)
print(
    "Simple-rule balanced accuracy:",
    round(
        balanced_accuracy_score(y_test, baseline_predictions),
        4,
    )
)

print("\nSimple-rule classification report:")
print(classification_report(
    y_test,
    baseline_predictions,
    labels=[0, 1],
    target_names=["Not low", "Low moisture"],
    zero_division=0,
))


# Compare with a simple majority-class baseline
baseline = int(y_train.mean() >= 0.5)
baseline_predictions = [baseline] * len(y_test)

print("Rows used:", len(df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Forecast horizon:", FORECAST_MINUTES, "minutes")
print("Provisional low-moisture threshold:", round(threshold, 4))

print("\nTest class proportions:")
print(y_test.value_counts(normalize=True).sort_index())

print("\nModel accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Model balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print(
    "Majority baseline accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4),
)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not low", "Low moisture"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save model and its metadata
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features,
        "forecast_minutes": FORECAST_MINUTES,
        "low_moisture_threshold": float(threshold),
        "label_definition": (
            "Future soil moisture at or below the training "
            "period's 20th percentile; provisional research label"
        ),
    },
    MODEL_PATH,
)

print("\nSaved model to:", MODEL_PATH)
