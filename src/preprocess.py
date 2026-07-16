import pandas as pd
import numpy as np
import urllib.request
import io
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET_COL = 'target'
RAW_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise induced angina (1=yes, 0=no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (1-3)",
    "ca": "Number of major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)"
}


def download_data(url: str = RAW_URL) -> pd.DataFrame:
    """Download and parse the UCI Heart Disease dataset."""
    all_cols = FEATURE_COLS + [TARGET_COL]
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode()
    df = pd.read_csv(io.StringIO(raw), header=None, names=all_cols, na_values='?')
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """Load pre-saved CSV."""
    return pd.read_csv(filepath)


def get_preprocessor() -> ColumnTransformer:
    """Return sklearn preprocessing pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer([('num', num_pipeline, FEATURE_COLS)])


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def get_missing_stats(df: pd.DataFrame) -> dict:
    """Return missing value counts per column."""
    return df.isnull().sum()[df.isnull().sum() > 0].to_dict()
