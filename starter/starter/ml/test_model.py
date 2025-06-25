import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from starter.starter.ml.model import train_model, inference, compute_model_metrics
from starter.starter.ml.data import process_data


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Load the census data once for all tests."""
    ROOT_DIR = Path(__file__).resolve().parent.parent 
    csv_path = ROOT_DIR / "data" / "census.csv"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def processed(data):
    """Return a processed feature matrix and label vector (sampled for speed)."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    subset = data.sample(n=500, random_state=42)
    X, y, _, _ = process_data(
        subset,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y


def test_train_model(processed) -> None:
    X, y = processed
    model = train_model(X, y)
    assert model.__class__.__name__ == "LogisticRegression"


def test_inference(processed) -> None:
    X, y = processed
    model = train_model(X, y)
    sample = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    preds = inference(model, pd.DataFrame([sample]))
    assert preds.shape == (1,), "Predictions should have same shape as (1,)"


def test_compute_model_metrics() -> None:
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1 and recall == 1 and fbeta == 1
