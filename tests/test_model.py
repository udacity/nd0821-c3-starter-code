import os

import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.model import compute_model_metrics, load_model, save_model, train_model


@pytest.mark.parametrize(
    "y, preds, expected_precision, expected_recall, expected_fbeta",
    [
        (np.array([1, 0, 1, 0, 1, 0]), np.array([1, 0, 1, 0, 1, 0]), 1.0, 1.0, 1.0),
        (np.array([1, 0, 1, 0, 1, 0]), np.array([0, 1, 0, 1, 0, 1]), 0.0, 0.0, 0.0),
        (
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([1, 1, 1, 0, 0, 0]),
            precision_score(
                np.array([1, 0, 1, 0, 1, 0]),
                np.array([1, 1, 1, 0, 0, 0]),
                zero_division=1,
            ),
            recall_score(
                np.array([1, 0, 1, 0, 1, 0]),
                np.array([1, 1, 1, 0, 0, 0]),
                zero_division=1,
            ),
            fbeta_score(
                np.array([1, 0, 1, 0, 1, 0]),
                np.array([1, 1, 1, 0, 0, 0]),
                beta=1,
                zero_division=1,
            ),
        ),
    ],
)
def test_compute_model_metrics(
    y, preds, expected_precision, expected_recall, expected_fbeta
):
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(expected_precision)
    assert recall == pytest.approx(expected_recall)
    assert fbeta == pytest.approx(expected_fbeta)


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=10, n_classes=2, random_state=42
    )
    return X, y


def test_train_model_returns_model(sample_data):
    x_train, y_train = sample_data
    model = train_model(x_train, y_train)

    assert is_classifier(model), "The returned object is not a classifier"
    assert hasattr(
        model, "predict"
    ), "The returned model does not have a predict method"


def test_model_can_predict(sample_data):
    x_train, y_train = sample_data
    model = train_model(x_train, y_train)
    predictions = model.predict(x_train)

    assert (
        predictions.shape[0] == x_train.shape[0]
    ), "The number of predictions does not match the number of samples"


def test_model_accuracy(sample_data):
    x_train, y_train = sample_data
    model = train_model(x_train, y_train)
    accuracy = model.score(x_train, y_train)

    assert accuracy > 0.8, "The model accuracy is lower than expected"


@pytest.fixture
def sample_model():
    model = RandomForestClassifier()
    encoder = StandardScaler()
    lb = LabelBinarizer()
    return model, encoder, lb


def test_save_model(tmpdir, sample_model):
    model, encoder, lb = sample_model
    filename = "test_model_save.pkl"
    MODEL_FOLDER = tmpdir

    save_model(model, encoder, lb, filename, folder=MODEL_FOLDER)
    saved_file_path = os.path.join(MODEL_FOLDER, filename)

    assert os.path.exists(saved_file_path), "Model file was not saved."


def test_load_model(tmpdir, sample_model):
    model, encoder, lb = sample_model
    filename = "test_model_load.pkl"
    MODEL_FOLDER = tmpdir

    save_model(model, encoder, lb, filename, folder=MODEL_FOLDER)
    loaded_model, loaded_encoder, loaded_lb = load_model(filename, folder=MODEL_FOLDER)

    assert isinstance(
        loaded_model, RandomForestClassifier
    ), "Loaded model is not of type RandomForestClassifier."
    assert isinstance(
        loaded_encoder, StandardScaler
    ), "Loaded encoder is not of type StandardScaler."
    assert isinstance(
        loaded_lb, LabelBinarizer
    ), "Loaded label binarizer is not of type LabelBinarizer."
