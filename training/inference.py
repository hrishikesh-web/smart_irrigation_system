"""
Smart Irrigation System - Unified Model Inference Backend
========================================================================
Loads trained models (Random Forest, LSTM, TCN, Transformer) and their 
corresponding metric files to provide live predictions and comparison tables 
for the Streamlit dashboard.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "training" / "models"


# ---------------------------------------------------------------------------
# PyTorch Model Architectures (matching training scripts)
# ---------------------------------------------------------------------------

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim=64, num_layers=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, n_features, channels=(64, 64, 128), kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_step = out[:, :, -1]
        return self.head(last_step)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Helper Loaders & Inference Wrappers
# ---------------------------------------------------------------------------

def get_metrics(model_name):
    mapping = {
        "Random Forest": "random_forest_metrics.json",
        "LSTM": "lstm_metrics.json",
        "TCN": "tcn_metrics.json",
        "Transformer": "transformer_metrics.json",
    }
    filename = mapping.get(model_name)
    if not filename:
        return None
    
    path = MODELS_DIR / filename
    if not path.exists():
        # Fallback check for alternate names
        if model_name == "LSTM":
            alt_path = MODELS_DIR / "improved_lstm_metrics.json"
            if alt_path.exists():
                path = alt_path
        if not path.exists():
            return None
            
    with open(path, "r") as f:
        return json.load(f)


def get_model_comparison_rows():
    rows = []
    models = ["Random Forest", "LSTM", "TCN", "Transformer"]
    for m in models:
        metrics = get_metrics(m)
        if metrics is not None:
            # Handle standard metrics dictionary structure
            if "validation_roc_auc" in metrics:
                roc = metrics.get("validation_roc_auc", 0)
                pr = metrics.get("validation_pr_auc", 0)
                thresh = metrics.get("optimal_threshold", 0.5)
                rows.append({
                    "Model": m,
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Status": "Trained & Active"
                })
            else:
                tuned = metrics.get("test_at_tuned_threshold", {})
                roc = tuned.get("roc_auc", 0)
                pr = tuned.get("pr_auc", 0)
                thresh = metrics.get("threshold_tuned", 0.5)
                rows.append({
                    "Model": m,
                    "ROC-AUC": f"{roc:.4f}",
                    "PR-AUC": f"{pr:.4f}",
                    "Tuned Threshold": f"{thresh:.3f}",
                    "Status": "Trained & Active"
                })
        else:
            rows.append({
                "Model": m,
                "ROC-AUC": "N/A",
                "PR-AUC": "N/A",
                "Tuned Threshold": "N/A",
                "Status": "Not Trained"
            })
    return rows


def predict(model_name, sensor_data, weather_data):
    """
    Executes live inference using the requested model name.
    """
    # Check if hardware / sensor readings are available
    if sensor_data.get("soil_moisture") is None:
        return {
            "required": None,
            "confidence": None,
            "message": f"Hardware not connected: live sensor data is N/A. Cannot run {model_name} prediction."
        }

    device = torch.device('cpu')
    
    # Handle Random Forest Baseline
    if model_name == "Random Forest":
        model_path = MODELS_DIR / "random_forest.pkl"
        if not model_path.exists():
            return {"required": None, "confidence": None, "message": "Random Forest model file not found."}
        rf_model = joblib.load(model_path)
        metrics = get_metrics("Random Forest")
        medians = metrics.get("feature_medians", {})
        
        # Build feature vector from medians + live sensors
        input_data = dict(medians)
        input_data["soil_moisture"] = sensor_data["soil_moisture"]
        if sensor_data.get("temperature") is not None:
            input_data["temperature"] = sensor_data["temperature"]
        if sensor_data.get("humidity") is not None:
            input_data["humidity"] = sensor_data["humidity"]
            
        features = list(metrics.get("features", input_data.keys()))
        row = [input_data.get(f, 0.0) for f in features]
        X_live = np.array([row])
        
        proba = rf_model.predict_proba(X_live)[0, 1]
        threshold = metrics.get("threshold_tuned", 0.5)
        required = bool(proba >= threshold)
        return {
            "required": required,
            "confidence": float(proba),
            "message": f"Random Forest prediction complete using threshold {threshold:.2f}."
        }

    # Handle Deep Learning Models (LSTM, TCN, Transformer)
    file_map = {
        "LSTM": ("improved_lstm.pt", ImprovedLSTMClassifier),
        "TCN": ("tcn.pt", TCNClassifiersPlaceholder if 'TCNClassifiersPlaceholder' in globals() else TCNClassifier),
        "Transformer": ("transformer.pt", TransformerClassifier)
    }
    
    if model_name not in file_map:
        return {"required": None, "confidence": None, "message": f"Unknown model type: {model_name}"}
        
    filename, model_cls = file_map[model_name]
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        # Fallback filename check
        if model_name == "LSTM" and (MODELS_DIR / "lstm.pt").exists():
            model_path = MODELS_DIR / "lstm.pt"
        else:
            return {"required": None, "confidence": None, "message": f"{model_name} weights file ({filename}) not found."}

    try:
        # Construct synthetic window sequence (24 steps) using live sensor reading
        n_features = 11  # Standard feature count
        model = model_cls(n_features=n_features).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict):
            try:
                model.load_state_dict(state_dict)
            except Exception:
                pass # Flexible loading fallback
        model.eval()

        # Create dummy 24-step sequence filled with current live sensor values
        dummy_window = np.zeros((1, 24, n_features), dtype=np.float32)
        dummy_window[0, :, 0] = sensor_data["soil_moisture"]
        if sensor_data.get("temperature") is not None:
            dummy_window[0, :, 1] = sensor_data["temperature"]
        if sensor_data.get("humidity") is not None:
            dummy_window[0, :, 2] = sensor_data["humidity"]

        with torch.no_grad():
            tensor_x = torch.tensor(dummy_window).to(device)
            logit = model(tensor_x)

        metrics = get_metrics(model_name)
        threshold = 0.5
        if metrics:
            threshold = metrics.get("optimal_threshold", metrics.get("threshold_tuned", 0.5))

        required = bool(proba >= threshold)
        return {
            "required": required,
            "confidence": float(proba),
            "message": f"{model_name} sequence evaluation complete at threshold {threshold:.2f}."
        }
    except Exception as e:
        return {
            "required": False,
            "confidence": 0.5,
            "message": f"Error running live inference for {model_name}: {str(e)}"
        }
