import os
import sys
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Make core/ importable when this script runs from managed_system_regression/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.energy import EnergyMeter

# Ensure directories exist
os.makedirs("knowledge", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load thresholds once for energy backend config
try:
    with open("knowledge/thresholds.json") as _tf:
        _thresholds = json.load(_tf)
except Exception:
    _thresholds = {}
_energy_backend = _thresholds.get("energy_meter", "auto")

# ---------------- Load Dataset ----------------
print("Loading data stream...")

df = pd.read_csv("knowledge/dataset.csv")
_value_col = _thresholds.get("value_column", "flow")
if _value_col not in df.columns:
    if "flow" in df.columns:
        _value_col = "flow"
    elif "value" in df.columns:
        _value_col = "value"
    else:
        raise KeyError(
            f"Column '{_value_col}' not found in knowledge/dataset.csv. "
            f"Available: {list(df.columns)}. "
            "Set value_column in thresholds.json or check preprocessing."
        )
data = df[_value_col].values

# B7 fix: load pre-fitted scaler from disk; never fit on full dataset here.
# Run scripts/init_regression.py once before starting inference to generate scaler.pkl.
scaler_path = "knowledge/scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(
        "knowledge/scaler.pkl not found. "
        "Run python scripts/init_regression.py to fit the scaler on training data."
    )
with open(scaler_path, "rb") as _f:
    scaler = pickle.load(_f)

data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 5
X_stream, y_stream = create_sequences(data_scaled, seq_length)

print("Data stream prepared. Streaming inference begins...")


# ---------------- Define LSTM Model ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# B4 fix: module-level model cache so we don't deserialise on every iteration.
# Reload happens only when model.csv changes or model_reload.flag is present.
_model_cache: dict = {}
_last_model_name: str = ""

RELOAD_FLAG = "knowledge/model_reload.flag"


def _load_model(model_name: str):
    """Load model from disk and store in cache. Returns the loaded model."""
    if model_name == "lstm":
        m = LSTMModel()
        m.load_state_dict(torch.load("models/lstm.pth", weights_only=False))
        m.eval()
        _model_cache["lstm"] = m
    elif model_name == "linear":
        with open("models/linear.pkl", "rb") as f:
            _model_cache["linear"] = pickle.load(f)
    elif model_name == "svm":
        with open("models/svm.pkl", "rb") as f:
            _model_cache["svm"] = pickle.load(f)
    return _model_cache.get(model_name)


def _get_model(model_name: str):
    """Return cached model, reloading only when necessary."""
    global _last_model_name

    force_reload = os.path.exists(RELOAD_FLAG)
    if force_reload:
        os.remove(RELOAD_FLAG)

    if model_name != _last_model_name or model_name not in _model_cache or force_reload:
        print(f"[cache] Loading model from disk: {model_name}")
        _load_model(model_name)
        _last_model_name = model_name

    return _model_cache.get(model_name)


# ---------------- Inference Loop ----------------
predictions_file = "knowledge/predictions.csv"
if not os.path.exists(predictions_file):
    pd.DataFrame(
        columns=["true_value", "predicted_value", "model_used", "inference_time", "energy_uJ"]
    ).to_csv(predictions_file, index=False)

for i in range(len(X_stream)):
    # Check active model
    try:
        with open("knowledge/model.csv", "r") as f:
            chosen_model = f.read().strip().lower()
    except FileNotFoundError:
        print("Error: knowledge/model.csv not found. Defaulting to lstm.")
        chosen_model = "lstm"

    print(f"Inference {i + 1}/{len(X_stream)}: Using model → {chosen_model.upper()}")

    X_input = X_stream[i].reshape(1, -1)

    with EnergyMeter("inference", backend=_energy_backend) as _em:
        start_time = time.time()

        model = _get_model(chosen_model)
        if model is None:
            print(f"Unknown model '{chosen_model}'. Defaulting to lstm.")
            chosen_model = "lstm"
            model = _get_model("lstm")

        if chosen_model == "lstm":
            X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(-1)
            prediction = model(X_tensor).detach().numpy().flatten()[0]
        else:
            prediction = model.predict(X_input)[0]

        inference_time = time.time() - start_time

    energy_usage_uJ = _em.total_uJ or 0.0

    true_value = y_stream[i]
    true_value_actual = scaler.inverse_transform([[true_value]])[0, 0]
    predicted_value_actual = scaler.inverse_transform([[prediction]])[0, 0]

    pd.DataFrame(
        [[true_value_actual, predicted_value_actual, chosen_model, inference_time, energy_usage_uJ]],
        columns=["true_value", "predicted_value", "model_used", "inference_time", "energy_uJ"],
    ).to_csv(predictions_file, mode="a", header=False, index=False)

    print(
        f"True: {true_value_actual:.2f}, Predicted: {predicted_value_actual:.2f}, "
        f"Model: {chosen_model.upper()}, Time: {inference_time:.6f}s, Energy: {energy_usage_uJ} µJ"
    )

    time.sleep(0.15)

print("\nStreaming inference completed. Predictions saved in knowledge/predictions.csv")
