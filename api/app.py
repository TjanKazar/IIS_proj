import os
import joblib
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Import Chronos utilities (adjust path if needed)
import sys
sys.path.append("/app/src")  # inside container
from chronos_utils import load_chronos_pipeline, forecast_all_metrics
from transforms import TargetEncoder  # only for class mapping

# ---------- Global loaded artifacts ----------
model = None
scaler = None
label_encoder = None
feature_cols = None
chronos_pipeline = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load MLP artifacts
    global model, scaler, label_encoder, feature_cols, chronos_pipeline
    model_path = "/app/models/mlp_traffic.pt"
    scaler_path = "/app/models/scaler.pkl"
    encoder_path = "/app/models/label_encoder.pkl"
    features_path = "/app/models/feature_cols.pkl"

    # Load MLP (assumes TrafficMLP class is defined)
    from train import TrafficMLP   # or duplicate class definition
    feature_cols = joblib.load(features_path)
    input_dim = len(feature_cols)
    num_classes = len(joblib.load(encoder_path).class_map_)
    model = TrafficMLP(input_dim=input_dim, hidden_dims=[256,128,64],
                       num_classes=num_classes, dropout_rate=0.3, batch_norm=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    # Optionally load Chronos (if GPU memory allows)
    try:
        chronos_pipeline = load_chronos_pipeline("amazon/chronos-bolt-small", device)
    except Exception as e:
        print(f"Chronos not loaded: {e}")
    yield
    # Cleanup
    if chronos_pipeline:
        del chronos_pipeline
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="Traffic State Predictor")

# ---------- Request/Response models ----------
class MetricsInput(BaseModel):
    volume: float = Field(..., description="St. vozil [N/h]")
    speed: float = Field(..., description="Hitrost [km/h]")
    headway: float = Field(..., description="Razmik [s]")
    occupancy: float = Field(..., description="Zasedenost [%]")

class ChronosInput(BaseModel):
    location: List[str] = Field(..., description="Location keys e.g. ['A1','Ljubljana','smer Koper']")
    context_len: int = Field(12, ge=2, le=48)
    prediction_len: int = Field(6, ge=1, le=12)

class PredictionResponse(BaseModel):
    predicted_stanje: str
    probabilities: Dict[str, float]
    forecasted_metrics: Optional[Dict[str, List[float]]] = None

# ---------- Helper: construct feature vector ----------
def build_feature_vector(volume, speed, headway, occupancy, last_row=None):
    """Create a feature vector for MLP. For now we just use the 4 metrics.
       In reality we'd need all features (lag, time features, etc.). 
       Simpler: the API expects the full preprocessed vector? 
       For demo we reconstruct from a template."""
    # Simplified: we assume the model was trained on only these 4 metrics
    # But your model uses many more (lag, time, categorical). 
    # For production you'd need to provide the full vector.
    # Here we cheat by loading a template from a stored "average" row and replace the 4 metrics.
    template = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if col == "St. vozil [N/h]":
            template[i] = volume
        elif col == "Hitrost [km/h]":
            template[i] = speed
        elif col == "Razmik [s]":
            template[i] = headway
        elif col == "Zasedenost [%]":
            template[i] = occupancy
    return template

@app.post("/predict", response_model=PredictionResponse)
async def predict(input: MetricsInput):
    """Direct classification from provided metrics."""
    feat = build_feature_vector(input.volume, input.speed, input.headway, input.occupancy)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).cpu().item())
    stanje = label_encoder.class_map_[pred_class]
    prob_dict = {label_encoder.class_map_[i]: float(probs[i]) for i in range(len(probs))}
    return PredictionResponse(predicted_stanje=stanje, probabilities=prob_dict)

@app.post("/forecast", response_model=PredictionResponse)
async def forecast_and_predict(input: ChronosInput):
    """Run Chronos forecast for the given location, then classify."""
    if chronos_pipeline is None:
        raise HTTPException(503, "Chronos model not available")
    # Load historical data (you'll need to mount a volume or database)
    # For simplicity, assume a CSV file is mounted at /app/data/raw.csv
    data_path = "/app/data/raw.csv"
    if not os.path.exists(data_path):
        raise HTTPException(404, "Historical data not found")
    df = pd.read_csv(data_path)
    # Filter to the requested location
    for i, col in enumerate(["Cesta", "Lokacija", "Smer"]):
        df = df[df[col].astype(str) == str(input.location[i])]
    df = df.sort_values("Čas")
    if len(df) < input.context_len:
        raise HTTPException(400, f"Not enough data for location, need {input.context_len} rows")
    # Run Chronos forecasts for all four metrics
    forecasts = forecast_all_metrics(
        pipeline=chronos_pipeline,
        location_df=df,
        target_columns=["St. vozil [N/h]", "Hitrost [km/h]", "Razmik [s]", "Zasedenost [%]"],
        context_len=input.context_len,
        prediction_len=input.prediction_len
    )
    # Extract the 30‑min forecast (last step)
    last_step = input.prediction_len - 1
    volume_f = forecasts["St. vozil [N/h]"][0][last_step]
    speed_f = forecasts["Hitrost [km/h]"][0][last_step]
    headway_f = forecasts["Razmik [s]"][0][last_step]
    occupancy_f = forecasts["Zasedenost [%]"][0][last_step]
    # Build feature vector and classify
    feat = build_feature_vector(volume_f, speed_f, headway_f, occupancy_f)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).cpu().item())
    stanje = label_encoder.class_map_[pred_class]
    prob_dict = {label_encoder.class_map_[i]: float(probs[i]) for i in range(len(probs))}
    # Return also full forecast trajectories
    forecast_metrics = {
        "volume": forecasts["St. vozil [N/h]"][0].tolist(),
        "speed": forecasts["Hitrost [km/h]"][0].tolist(),
        "headway": forecasts["Razmik [s]"][0].tolist(),
        "occupancy": forecasts["Zasedenost [%]"][0].tolist(),
    }
    return PredictionResponse(predicted_stanje=stanje, probabilities=prob_dict,
                              forecasted_metrics=forecast_metrics)

@app.get("/health")
async def health():
    return {"status": "ok"}