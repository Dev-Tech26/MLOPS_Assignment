"""Unit tests for model training and evaluation."""
import pytest
import sys, os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.preprocess import get_preprocessor, FEATURE_COLS, TARGET_COL
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SAMPLE_DATA = {
    'age': [63,37,41,56,57,60,45,52,62,48]*3,
    'sex': [1,1,0,1,0,1,0,1,0,1]*3,
    'cp': [1,3,2,2,2,1,3,2,4,2]*3,
    'trestbps': [145,130,130,120,120,140,110,130,150,125]*3,
    'chol': [233,250,204,236,354,200,220,240,260,210]*3,
    'fbs': [1,0,0,0,0,1,0,0,1,0]*3,
    'restecg': [2,0,2,0,0,1,0,2,0,1]*3,
    'thalach': [150,187,172,178,163,155,160,170,140,165]*3,
    'exang': [0,0,0,0,1,0,1,0,1,0]*3,
    'oldpeak': [2.3,3.5,1.4,0.8,0.6,1.0,0.5,1.5,2.0,0.8]*3,
    'slope': [3,3,1,1,1,2,1,2,3,1]*3,
    'ca': [0,0,0,0,0,1,0,2,1,0]*3,
    'thal': [6,3,3,3,3,7,3,7,3,3]*3,
    'target': ([0]*10 + [1]*10 + [0]*10)
}


@pytest.fixture
def sample_df():
    return pd.DataFrame(SAMPLE_DATA)


def test_pipeline_fits_and_predicts(sample_df):
    pipe = Pipeline([("pre", get_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    X, y = sample_df[FEATURE_COLS], sample_df[TARGET_COL]
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_model_has_predict_proba(sample_df):
    pipe = Pipeline([("pre", get_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    X, y = sample_df[FEATURE_COLS], sample_df[TARGET_COL]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_roc_auc_above_chance(sample_df):
    pipe = Pipeline([("pre", get_preprocessor()), ("clf", LogisticRegression(max_iter=1000))])
    X, y = sample_df[FEATURE_COLS], sample_df[TARGET_COL]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    assert auc >= 0.5, f"AUC {auc} is below chance level"


def test_preprocessor_is_sklearn_transformer(sample_df):
    from sklearn.utils.estimator_checks import parametrize_with_checks
    pre = get_preprocessor()
    X = sample_df[FEATURE_COLS]
    pre.fit(X)
    out = pre.transform(X)
    assert out.shape[1] == len(FEATURE_COLS)
