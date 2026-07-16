# Heart Disease MLOps Project

> End-to-End ML Pipeline: Data → EDA → Model Training → API → Docker → Kubernetes → CI/CD → Monitoring

[![CI/CD](https://github.com/your-org/heart-disease-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/heart-disease-mlops/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

## Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)  
- **Processing**: Cleveland subset (303 patients, 13 features, binary target)
- **Download**: `python src/download_data.py`

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-org/heart-disease-mlops.git
cd heart-disease-mlops
pip install -r requirements.txt

# 2. Download data
python src/download_data.py

# 3. EDA
python src/eda.py

# 4. Train models (with MLflow tracking)
python src/train.py

# 5. Run API (local)
uvicorn src.predict:app --host 0.0.0.0 --port 8000 --reload

# 6. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":3,"trestbps":125,"chol":212,"fbs":0,
       "restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'

# 7. Run tests
pytest tests/ -v --cov=src

# 8. MLflow UI
mlflow ui --port 5000
```

## Docker

```bash
# Build
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models heart-disease-api:latest

# Full stack (API + Prometheus + Grafana)
docker-compose -f docker/docker-compose.yml up
```

## Kubernetes (Minikube)

```bash
minikube start
minikube image load heart-disease-api:latest
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
minikube service heart-disease-api-svc -n mlops --url
```

## Helm

```bash
helm install heart-disease ./helm/heart-disease --namespace mlops --create-namespace
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/docs` | Swagger UI |
| GET | `/metrics` | Prometheus metrics |

## Sample Request

```json
POST /predict
{
  "age": 52, "sex": 1, "cp": 3, "trestbps": 125,
  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

## Sample Response

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "confidence": 0.8732,
  "risk_level": "High",
  "model_version": "1.0.0"
}
```

## Project Structure

```
heart-disease-mlops/
├── data/               # Dataset files
├── src/
│   ├── download_data.py   # Dataset downloader
│   ├── preprocess.py      # Preprocessing pipeline
│   ├── eda.py             # EDA script
│   ├── train.py           # Model training + MLflow
│   └── predict.py         # FastAPI serving app
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_api.py
├── models/             # Saved model artifacts
├── reports/            # EDA plots & charts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── helm/heart-disease/ # Helm chart
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── grafana/
├── .github/workflows/
│   └── ci-cd.yml       # GitHub Actions pipeline
└── requirements.txt
```

## Technology Stack

| Category | Tool |
|----------|------|
| Language | Python 3.11 |
| ML | scikit-learn, GradientBoosting |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| Orchestration | Kubernetes / Minikube |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / admin123)

### Key Metrics Tracked

- `heart_api_requests_total` — Total request count by status
- `heart_api_latency_seconds` — Request latency histogram
- `heart_predictions_total` — Predictions by class (0/1)
- `heart_confidence_score` — Distribution of confidence scores
