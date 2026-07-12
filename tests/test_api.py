"""Unit tests for FastAPI prediction endpoints."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# We test the app without loading a real model
from unittest.mock import patch, MagicMock
import numpy as np

SAMPLE_PATIENT = {
    "age": 52, "sex": 1, "cp": 3, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}


@pytest.fixture
def client_with_model():
    """Create test client with a mock model."""
    from fastapi.testclient import TestClient
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("predict.model", mock_model):
        import predict
        predict.model = mock_model
        from fastapi.testclient import TestClient
        client = TestClient(predict.app)
        yield client


def test_health_endpoint(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_root_endpoint(client_with_model):
    resp = client_with_model.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_predict_structure(client_with_model):
    resp = client_with_model.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "prediction_label" in data
    assert "risk_level" in data


def test_predict_output_types(client_with_model):
    resp = client_with_model.post("/predict", json=SAMPLE_PATIENT)
    data = resp.json()
    assert isinstance(data["prediction"], int)
    assert isinstance(data["confidence"], float)
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_invalid_age(client_with_model):
    bad = {**SAMPLE_PATIENT, "age": -5}
    resp = client_with_model.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_missing_field(client_with_model):
    bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    resp = client_with_model.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint(client_with_model):
    resp = client_with_model.get("/metrics")
    assert resp.status_code == 200
