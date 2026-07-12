# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY src/predict.py ./src/predict.py
COPY src/preprocess.py ./src/preprocess.py
COPY src/__init__.py ./src/__init__.py
COPY models/best_model.joblib ./models/best_model.joblib

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Env
ENV MODEL_PATH=models/best_model.joblib
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
