from typing import Any
from pytest import fixture
from data import process_data, import_data 
from model import train_model, compute_model_metrics, inference
from sklearn.datasets import make_classification
from yaml import load, Loader
import pandas as pd
import numpy as np

from starter.starter.train_model import X_test, X_train
paths_dict = load(open("starter/path.yml"), Loader=Loader)
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
    return data


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
    y = data['salary']
    preds = data['salary']*1.5
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "Fbeta should be a float."

@fixture(scope='session')
def test_train_model(data):
    """ Test that the model is valid.
    Inpputs
    -------
    data : pd.DataFrame
        Dataframe containing the data
    
    Returns
    -------
    model : Any
        Model object.
    X_test : np.ndarray
        Test data.
    """
    X, y = make_classification(n_samples=400, n_features=2, n_informative=2, n_redundant=0, random_state=0)
    X_train, y_train, X_test, y_test = X[:300], y[:300], X[300:], y[300:]
    model = train_model(X_train, y_train)

    assert model is not None, "Model should not be None."

    return model, X_test
    

def test_inference(model: Any, X_test: np.ndarray):
    """ Test that the inference is valid.
    Inpputs
    -------
    model : Any
        Model object.
    X_test : np.ndarray
        Test data.
    """
    predictions = inference(model, X_test)

    assert type(predictions) is np.ndarray, "Predictions should be a numpy array."
    assert predictions.shape[0] == X_test.shape[0], "Predictions should have shape equal to test data shape."
    
    
    
    
    
    
    
    
    
    
    