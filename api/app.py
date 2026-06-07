"""
app.py
──────
FastAPI endpoint for traffic state prediction.
Uses trained MLP classifier + LSTM forecasters.
Designed for deployment on Render free tier (512 MB RAM).
"""

import os
import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# ── Path setup ────────────────────────────────────────────────────────────
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine project root
if os.path.isdir(os.path.join(API_DIR, "models")):
    PROJECT_ROOT = API_DIR
else:
    PROJECT_ROOT = os.path.dirname(API_DIR)

# Add ALL necessary paths BEFORE any imports that might need them
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "model"))

# Now imports work
from src.model.lstm_utils import load_all_forecasters

# ── Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "traffic.csv")

MODELS_DIR = os.environ.get("MODELS_DIR", MODELS_DIR)
DATA_PATH = os.environ.get("DATA_PATH", DATA_PATH)

print(f"[api] Project root : {PROJECT_ROOT}")
print(f"[api] Models dir   : {MODELS_DIR}")
print(f"[api] Data path    : {DATA_PATH}")
print(f"[api] sys.path     : {sys.path[:3]}")

# ── TrafficMLP class ──────────────────────────────────────────────────────
class TrafficMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int,
                 dropout_rate: float = 0.3, batch_norm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ── Global state ──────────────────────────────────────────────────────────
model = None
scaler = None
label_encoder = None
feature_cols = None
forecasters = None
device = torch.device("cpu")

COL_VOLUME    = "Št. vozil [N/h]"
COL_SPEED     = "Hitrost [km/h]"
COL_HEADWAY   = "Razmik [s]"
COL_OCCUPANCY = "Zasedenost [%]"
FORECAST_METRICS = [COL_VOLUME, COL_SPEED, COL_HEADWAY, COL_OCCUPANCY]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, label_encoder, feature_cols, forecasters

    print("[api] Loading models...")
    
    features_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature columns not found at {features_path}")
    feature_cols = joblib.load(features_path)
    print(f"[api] Features: {len(feature_cols)} columns")

    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
    label_encoder = joblib.load(encoder_path)
    print(f"[api] Classes: {len(label_encoder.class_map_)}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("[api] Scaler loaded")

    model_path = os.path.join(MODELS_DIR, "mlp_traffic.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MLP model not found at {model_path}")
    
    input_dim = len(feature_cols)
    num_classes = len(label_encoder.class_map_)
    
    model = TrafficMLP(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=num_classes,
        dropout_rate=0.3,
        batch_norm=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[api] MLP loaded: {sum(p.numel() for p in model.parameters()):,} params")

    try:
        forecasters = load_all_forecasters(
            FORECAST_METRICS,
            context_len=12,
            pred_len=6,
            models_dir=MODELS_DIR
        )
        print(f"[api] Forecasters loaded: {list(forecasters.keys())}")
    except Exception as e:
        print(f"[api] WARNING: Forecasters not loaded: {e}")
        forecasters = None

    print("[api] Ready.")
    yield
    print("[api] Shutting down...")


app = FastAPI(lifespan=lifespan, title="Traffic State Predictor")


# ── Pydantic models ───────────────────────────────────────────────────────
class MetricsInput(BaseModel):
    volume: float = Field(..., description="Št. vozil [N/h]")
    speed: float = Field(..., description="Hitrost [km/h]")
    headway: float = Field(..., description="Razmik [s]")
    occupancy: float = Field(..., description="Zasedenost [%]")
    
    class Config:
        json_schema_extra = {
            "example": {"volume": 1200.5, "speed": 85.3, "headway": 3.2, "occupancy": 12.7}
        }

class ForecastInput(BaseModel):
    location: List[str] = Field(..., description="Location keys e.g. ['A1','Ljubljana','smer Koper']")
    context_len: int = Field(12, ge=6, le=48)
    prediction_len: int = Field(6, ge=1, le=12)
    
    class Config:
        json_schema_extra = {
            "example": {"location": ["A1","Ljubljana","smer Koper"], "context_len": 12, "prediction_len": 6}
        }

class PredictionResponse(BaseModel):
    predicted_stanje: str
    probabilities: Dict[str, float]
    forecasted_metrics: Optional[Dict[str, List[float]]] = None
    forecast_step_minutes: Optional[List[int]] = None
    location: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────
def build_feature_vector(volume, speed, headway, occupancy, template_row=None):
    if template_row is not None:
        feat = template_row.copy()
        for col, val in [(COL_VOLUME, volume), (COL_SPEED, speed),
                         (COL_HEADWAY, headway), (COL_OCCUPANCY, occupancy)]:
            if col in feature_cols:
                feat[feature_cols.index(col)] = val
        return feat
    else:
        feat = np.zeros(len(feature_cols), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            if col == COL_VOLUME: feat[i] = volume
            elif col == COL_SPEED: feat[i] = speed
            elif col == COL_HEADWAY: feat[i] = headway
            elif col == COL_OCCUPANCY: feat[i] = occupancy
        return feat

def classify(feat_vector):
    feat_scaled = scaler.transform(feat_vector.reshape(1, -1))
    tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).cpu().item())
    stanje = label_encoder.class_map_[pred_class]
    prob_dict = {label_encoder.class_map_[i]: float(probs[i]) for i in range(len(probs))}
    return stanje, prob_dict


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mlp_loaded": model is not None,
        "forecasters_loaded": forecasters is not None,
        "data_exists": os.path.exists(DATA_PATH)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input: MetricsInput):
    feat = build_feature_vector(input.volume, input.speed, input.headway, input.occupancy)
    stanje, prob_dict = classify(feat)
    return PredictionResponse(predicted_stanje=stanje, probabilities=prob_dict)

@app.post("/forecast", response_model=PredictionResponse)
async def forecast_and_predict(input: ForecastInput):
    if forecasters is None:
        raise HTTPException(503, "Forecasters not loaded")
    if not os.path.exists(DATA_PATH):
        raise HTTPException(404, f"Data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    possible_cols = [
        ["Cesta", "Lokacija", "Smer"],
        ["cesta", "lokacija", "smer"],
        ["road", "location", "direction"],
    ]
    location_cols = None
    for candidate in possible_cols:
        if all(c in df.columns for c in candidate):
            location_cols = candidate
            break
    
    if location_cols is None:
        str_cols = [c for c in df.columns if df[c].dtype == object][:3]
        if len(str_cols) >= len(input.location):
            location_cols = str_cols[:len(input.location)]
        else:
            raise HTTPException(400, f"Cannot determine location columns. Available: {list(df.columns)}")
    
    for col, val in zip(location_cols, input.location):
        df = df[df[col].astype(str) == str(val)]
    
    if len(df) == 0:
        raise HTTPException(404, f"No data for location {input.location}")
    
    time_cols = ["Čas", "čas", "Timestamp", "timestamp", "DATUM"]
    for tc in time_cols:
        if tc in df.columns:
            df = df.sort_values(tc)
            break
    
    if len(df) < input.context_len:
        raise HTTPException(400, f"Need {input.context_len} rows, got {len(df)}")
    
    recent_df = df.tail(input.context_len)
    
    forecasts = {}
    for metric in FORECAST_METRICS:
        if metric not in df.columns:
            continue
        series = recent_df[metric].dropna().values.astype(np.float32)
        if len(series) < input.context_len:
            continue
        ctx = torch.tensor(series[-input.context_len:], dtype=torch.float32)
        mean_fc, q10_fc, q90_fc = forecasters[metric].predict(ctx)
        pred_len = min(input.prediction_len, len(mean_fc))
        forecasts[metric] = {"mean": mean_fc[:pred_len], "q10": q10_fc[:pred_len], "q90": q90_fc[:pred_len]}
    
    v = forecasts[COL_VOLUME]["mean"][-1] if COL_VOLUME in forecasts else 0
    s = forecasts[COL_SPEED]["mean"][-1] if COL_SPEED in forecasts else 0
    h = forecasts[COL_HEADWAY]["mean"][-1] if COL_HEADWAY in forecasts else 0
    o = forecasts[COL_OCCUPANCY]["mean"][-1] if COL_OCCUPANCY in forecasts else 0
    
    feat = build_feature_vector(v, s, h, o)
    stanje, prob_dict = classify(feat)
    
    forecast_metrics = {m: forecasts[m]["mean"].tolist() for m in forecasts}
    steps = [5 * (i + 1) for i in range(min(input.prediction_len, 6))]
    
    return PredictionResponse(
        predicted_stanje=stanje,
        probabilities=prob_dict,
        forecasted_metrics=forecast_metrics if forecast_metrics else None,
        forecast_step_minutes=steps,
        location=" → ".join(input.location)
    )

@app.get("/info")
async def info():
    return {
        "project_root": PROJECT_ROOT,
        "models_dir": MODELS_DIR,
        "data_path": DATA_PATH,
        "data_exists": os.path.exists(DATA_PATH),
        "num_features": len(feature_cols) if feature_cols else 0,
        "classes": label_encoder.class_map_ if label_encoder else None,
        "forecasters_available": list(forecasters.keys()) if forecasters else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)