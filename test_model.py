from sklearn.linear_model import LogisticRegression

from starter.starter.ml.model import load_model
from starter.starter.ml.model import train_model, inference
from starter.starter.ml.model import compute_model_metrics, compute_slice_metrics

import pandas as pd
import pytest

@pytest.fixture
def test_load_model(root_path):
    model = load_model(root_path, "model.pkl")

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 300
    assert model.n_features_in_ == 108


def test_train_model(data):
    X, y = data
    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 300
    assert model.n_features_in_ == 4


def test_inference(model, data):
    X, y = data
    y_pred = inference(model, X)

    assert len(y_pred) == len(y)
    assert y_pred.any() == 1


def test_compute_model_metrics(model, data):
    X, y = data
    y_pred = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    assert precision > 0.0
    assert recall > 0.0
    assert fbeta > 0.0


def test_compute_slice_metrics(data):
    X, y = data
    slice_performance = compute_slice_metrics(
        features=X,
        labels=y,
        predictions=y,
        cat_features=['A']
    )

    assert isinstance(slice_performance, pd.DataFrame)
    assert (slice_performance.columns == ['Precision', 'Recall', 'TNR', 'NPV', 'F-Score']).all()