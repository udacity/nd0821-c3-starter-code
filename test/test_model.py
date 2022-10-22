from typing import Any
from pytest import fixture
from starter.ml.data import import_data
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.datasets import make_classification
from yaml import load, Loader
import pandas as pd
import numpy as np

paths_dict = load(open("path.yml"), Loader=Loader)
DATA_PATH = paths_dict["DATA_PATH"]


@fixture(scope='session')
def data() -> pd.DataFrame:
    """ Load the data.

    Returns
    -------
    data : pd.DataFrame
        Dataframe containing the data.
    """
    data = import_data(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


@fixture(scope='session')
def model(data: pd.DataFrame) -> Any:
    """ Train and save ML model create to predict the salary of a person based on their demographic features.
    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the data.

    Returns
    -------
    model : Any
        Model object.
    """
    X, y = make_classification(
        n_samples=400, n_features=2, n_informative=2, n_redundant=0, random_state=0)
    model = train_model(X, y)

    assert model is not None, "Model should not be None."

    return model


def test_data_validity(data: pd.DataFrame):
    """ Test that the data is valid. """
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_compute_model_metrics(data: pd.DataFrame):
    """ Test that the model metrics are valid.
    Inpputs
    -------
    data : pd.DataFrame
        Dataframe containing the data.
    """
    y = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    preds = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "Fbeta should be a float."


def test_inference(model: Any):
    """ Test that the inference is valid.
    Inpputs
    -------
    model : Any
        Model object.
    X_test : np.ndarray
        Test data.
    """
    X_test, _ = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)
    predictions = inference(model, X_test)

    assert isinstance(
        predictions, np.ndarray), "Predictions should be a numpy array."
    assert predictions.shape[0] == X_test.shape[0], "Predictions should have shape equal to test data shape."
