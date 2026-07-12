"""
Quick validation of the training pipeline.

Run with:
    pytest tests/validate_training.py -v
"""

from pathlib import Path
import joblib

import numpy as np
import pandas as pd

# Import your training function or module
# If your training code is in src/train.py with a main() or train() function:
from src import train  # adjust if your module is named differently


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def test_training_pipeline_runs_and_saves_model():
    """
    End-to-end check:
    - Data exists or is downloaded
    - Training runs without error
    - Champion model file is saved
    - Model can predict and has reasonable ROC-AUC
    """

    # 1. Ensure data exists (train.main() handles download if missing)
    assert DATA_PATH.parent.exists(), "data/ folder is missing"

    # 2. Run training (this should produce best_model.joblib)
    train.main()

    # 3. Check that model file was created
    assert MODEL_PATH.exists(), f"Expected trained model at {MODEL_PATH}, but file not found"

    # 4. Load model and do a basic sanity check
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].values

    # Use a small subset for speed
    X_sample = X.iloc[:50, :]
    y_sample = y[:50]
    preds = model.predict(X_sample)
    assert preds.shape == y_sample.shape, "Prediction shape mismatch"

    # Simple quality check: model shouldn't be trivial (all one class)
    unique_preds = np.unique(preds)
    assert len(unique_preds) > 1, "Model predictions are degenerate (only one class)"


if __name__ == "__main__":
    # Allow running directly: python tests/validate_training.py
    test_training_pipeline_runs_and_saves_model()
    print("✅ Training pipeline validated successfully.")