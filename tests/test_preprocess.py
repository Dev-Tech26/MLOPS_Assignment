"""Unit tests for data preprocessing."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.preprocess import get_preprocessor, split_data, get_missing_stats, FEATURE_COLS, TARGET_COL


SAMPLE_DATA = {
    'age': [63, 37, 41, 56, 57],
    'sex': [1, 1, 0, 1, 0],
    'cp': [1, 3, 2, 2, 2],
    'trestbps': [145, 130, 130, 120, 120],
    'chol': [233, 250, 204, 236, 354],
    'fbs': [1, 0, 0, 0, 0],
    'restecg': [2, 0, 2, 0, 0],
    'thalach': [150, 187, 172, 178, 163],
    'exang': [0, 0, 0, 0, 1],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
    'slope': [3, 3, 1, 1, 1],
    'ca': [0, 0, 0, 0, 0],
    'thal': [6, 3, 3, 3, 3],
    'target': [0, 0, 0, 0, 0],
}


@pytest.fixture
def sample_df():
    return pd.DataFrame(SAMPLE_DATA)


def test_feature_columns_present(sample_df):
    for col in FEATURE_COLS:
        assert col in sample_df.columns, f"Missing feature column: {col}"


def test_target_column_present(sample_df):
    assert TARGET_COL in sample_df.columns


def test_target_binary(sample_df):
    assert set(sample_df[TARGET_COL].unique()).issubset({0, 1})


def test_preprocessor_output_shape(sample_df):
    pre = get_preprocessor()
    X = sample_df[FEATURE_COLS]
    X_transformed = pre.fit_transform(X)
    assert X_transformed.shape == (len(sample_df), len(FEATURE_COLS))


def test_preprocessor_handles_missing():
    df_missing = pd.DataFrame(SAMPLE_DATA)
    df_missing.loc[0, 'ca'] = np.nan
    df_missing.loc[1, 'thal'] = np.nan
    pre = get_preprocessor()
    X_out = pre.fit_transform(df_missing[FEATURE_COLS])
    assert not np.isnan(X_out).any(), "Preprocessor should fill NaN values"


def test_split_proportions(sample_df):
    # Need at least 10 samples for stratified split
    df_big = pd.concat([sample_df] * 10, ignore_index=True)
    df_big['target'] = [0]*25 + [1]*25
    X_train, X_test, y_train, y_test = split_data(df_big, test_size=0.2)
    total = len(df_big)
    assert abs(len(X_test) / total - 0.2) < 0.1


def test_missing_stats(sample_df):
    sample_df_copy = sample_df.copy()
    sample_df_copy.loc[0, 'ca'] = np.nan
    stats = get_missing_stats(sample_df_copy)
    assert 'ca' in stats
    assert stats['ca'] == 1


def test_no_data_leakage(sample_df):
    df_big = pd.concat([sample_df] * 10, ignore_index=True)
    df_big['target'] = [0]*25 + [1]*25
    X_train, X_test, _, _ = split_data(df_big)
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(test_idx), "Train and test sets must not overlap"
