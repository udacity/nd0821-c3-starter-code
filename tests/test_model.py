import numpy as np
from starter.ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier


def test_train_model(data_train):
    X_train, y_train = data_train
    model = train_model(X_train, y_train)
    assert type(model) == RandomForestClassifier


def test_inference(model, data_test):
    X_test, y = data_test
    prediction = inference(model, X_test)
    assert type(prediction) == np.ndarray
    assert len(y) == len(prediction)


def test_compute_model_metrics(data_test, predictions):
    _, y_test = data_test
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
