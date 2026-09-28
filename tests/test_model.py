from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_artifact,
    train_model,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

CAT_FEATURES = ["workclass", "sex"]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "hours-per-week": rng.integers(10, 60, n),
        "workclass": rng.choice(["Private", "State-gov", "Self-emp-inc"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "salary": rng.choice(["<=50K", ">50K"], n),
    })


@pytest.fixture
def processed(data):
    X, y, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y, encoder, lb


def test_process_data_shapes_and_labels(data, processed):
    X, y, encoder, lb = processed
    # 2 continuous columns + 3 workclass + 2 sex one-hot columns.
    assert X.shape == (len(data), 7)
    assert y.shape == (len(data),)
    assert set(np.unique(y)) <= {0, 1}
    assert list(lb.classes_) == ["<=50K", ">50K"]


def test_process_data_inference_reuses_encoder(data, processed):
    X_train, _, encoder, lb = processed
    X, y, encoder_out, lb_out = process_data(
        data, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )
    assert encoder_out is encoder
    assert lb_out is lb
    np.testing.assert_array_equal(X, X_train)


def test_train_model_returns_fitted_random_forest(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_features_in_ == X.shape[1]


def test_inference_returns_binary_prediction_per_row(processed):
    X, y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)


def test_compute_slice_metrics_one_row_per_value(data, processed):
    X, y, encoder, lb = processed
    model = train_model(X, y)
    slices = compute_slice_metrics(
        model, data, "workclass", CAT_FEATURES, "salary", encoder, lb
    )
    assert sorted(slices["value"]) == sorted(data["workclass"].unique())
    assert slices["n"].sum() == len(data)
    assert slices[["precision", "recall", "fbeta"]].apply(lambda s: s.between(0, 1)).all().all()


def test_saved_artifacts_run_inference():
    model = load_artifact(MODEL_DIR / "model.joblib")
    encoder = load_artifact(MODEL_DIR / "encoder.joblib")
    lb = load_artifact(MODEL_DIR / "lb.joblib")
    sample = pd.read_csv(ROOT / "data" / "census.csv").head(10)
    X, _, _, _ = process_data(
        sample.drop(columns=["salary"]),
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ],
        training=False, encoder=encoder, lb=lb,
    )
    preds = inference(model, X)
    assert preds.shape == (10,)
    assert set(lb.inverse_transform(preds)) <= {"<=50K", ">50K"}
