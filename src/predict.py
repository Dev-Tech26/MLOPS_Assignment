import os
import time
import logging
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)


def _resolve_model_path():
    candidates = []
    if "__file__" in globals():
        notebook_dir = Path(__file__).resolve().parent
        candidates.extend([
            notebook_dir / "models" / "best_model.joblib",
            notebook_dir.parent / "models" / "best_model.joblib",
        ])
    candidates.extend([
        Path.cwd() / "models" / "best_model.joblib",
        Path.cwd().parent / "models" / "best_model.joblib",
    ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return os.getenv("MODEL_PATH", "models/best_model.joblib")


MODEL_PATH = _resolve_model_path()

# ── Prometheus metrics ──────────────────────────────────────────────────────────
METRICS_REGISTRY = CollectorRegistry(auto_describe=True)
REQUEST_COUNT = Counter("heart_api_requests_total", "Total API requests", ["method", "endpoint", "status"], registry=METRICS_REGISTRY)
REQUEST_LATENCY = Histogram("heart_api_latency_seconds", "Request latency", ["endpoint"], registry=METRICS_REGISTRY)
PREDICTION_COUNTER = Counter("heart_predictions_total", "Predictions", ["prediction"], registry=METRICS_REGISTRY)
CONFIDENCE_HISTOGRAM = Histogram("heart_confidence_score", "Confidence scores", buckets=[.3, .5, .6, .7, .8, .9, 1.0], registry=METRICS_REGISTRY)

# ── Load model at startup ──────────────────────────────────────────────────────
model = None


def _get_model():
    global model
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_model()
    yield


app = FastAPI(
    title="Heart Disease Risk Predictor",
    description="MLOps API for Heart Disease classification using the UCI dataset",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years", json_schema_extra={"example": 52})
    sex: float = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)", json_schema_extra={"example": 1})
    cp: float = Field(..., ge=1, le=4, description="Chest pain type", json_schema_extra={"example": 3})
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mmHg)", json_schema_extra={"example": 125})
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)", json_schema_extra={"example": 212})
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 (1=yes)", json_schema_extra={"example": 0})
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results", json_schema_extra={"example": 1})
    thalach: float = Field(..., ge=60, le=250, description="Max heart rate achieved", json_schema_extra={"example": 168})
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression", json_schema_extra={"example": 1.0})
    slope: float = Field(..., ge=1, le=3, description="Slope of peak ST segment", json_schema_extra={"example": 2})
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels (0-3)", json_schema_extra={"example": 2})
    thal: float = Field(..., description="Thalassemia (3=normal, 6=fixed, 7=reversible)", json_schema_extra={"example": 3})


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    risk_level: str
    model_version: str = "1.0.0"


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}


# ── Prometheus metrics endpoint ────────────────────────────────────────────────
@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(registry=METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {"message": "Heart Disease Risk Predictor API", "docs": "/docs", "health": "/health"}


# ── Predict ────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientData, request: Request):
    active_model = _get_model()
    if active_model is None:
        REQUEST_COUNT.labels("POST", "/predict", "503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    start = time.time()
    try:
        feature_order = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                         "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        X = pd.DataFrame([patient.model_dump()])[feature_order]
        pred = int(active_model.predict(X)[0])
        prob = float(active_model.predict_proba(X)[0][1])
        confidence = prob if pred == 1 else 1 - prob

        label = "Heart Disease" if pred == 1 else "No Heart Disease"
        if prob >= 0.75:
            risk = "High"
        elif prob >= 0.5:
            risk = "Moderate"
        else:
            risk = "Low"

        PREDICTION_COUNTER.labels(str(pred)).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        REQUEST_COUNT.labels("POST", "/predict", "200").inc()
        REQUEST_LATENCY.labels("/predict").observe(time.time() - start)

        logger.info(json.dumps({
            "event": "prediction",
            "prediction": pred,
            "confidence": round(confidence, 4),
            "risk": risk,
            "latency_ms": round((time.time() - start) * 1000, 2),
        }))

        return PredictionResponse(
            prediction=pred,
            prediction_label=label,
            confidence=round(confidence, 4),
            risk_level=risk,
        )
    except Exception as e:
        REQUEST_COUNT.labels("POST", "/predict", "500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch predict ──────────────────────────────────────────────────────────────
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientData]):
    active_model = _get_model()
    if active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    feature_order = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    X = pd.DataFrame([p.model_dump() for p in patients])[feature_order]
    preds = active_model.predict(X).tolist()
    probs = active_model.predict_proba(X)[:, 1].tolist()
    return [{"index": i, "prediction": int(p), "probability": round(pr, 4)}
            for i, (p, pr) in enumerate(zip(preds, probs))]


# Example: run a sample request and print the prediction output
sample_patient = {
    "age": 52,
    "sex": 1,
    "cp": 3,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}

active_model = _get_model()
if active_model is None:
    print("Model could not be loaded. Check the model path.")
else:
    client = TestClient(app)
    response = client.post("/predict", json=sample_patient)
    print(response.status_code)
    print(response.json())