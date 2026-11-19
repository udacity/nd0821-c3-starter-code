import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

CATS = ["color", "shape"]


@pytest.fixture
def arrays():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


@pytest.fixture
def df():
    rng = np.random.RandomState(1)
    n = 200
    d = pd.DataFrame(
        {
            "num": rng.rand(n),
            "color": rng.choice(["red", "green", "blue"], n),
            "shape": rng.choice(["sq", "circle"], n),
        }
    )
    d["salary"] = np.where(d["num"] > 0.5, ">50K", "<=50K")
    return d


def test_train_model_returns_fitted_random_forest(arrays):
    X, y = arrays
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_shape_and_binary_values(arrays):
    X, y = arrays
    preds = inference(train_model(X, y), X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_compute_slice_metrics_one_row_per_category(df):
    X, y, enc, lb = process_data(df, CATS, label="salary", training=True)
    model = train_model(X, y)
    out = compute_slice_metrics(df, "color", model, CATS, enc, lb)
    assert len(out) == df["color"].nunique()
    assert out["n"].sum() == len(df)
    assert (
        out[["precision", "recall", "fbeta"]]
        .apply(lambda c: c.between(0, 1))
        .all()
        .all()
    )
