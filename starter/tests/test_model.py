import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

import starter.starter.ml.model as model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_can_train_model(get_process_data):
    X, y, encoder, lb = get_process_data
    ml_model = model.train_model(X_train=X, y_train=y)
    assert ml_model is not None
    assert isinstance(ml_model, LogisticRegression)


def test_can_compute_model_metrics(get_process_data):
    X, y, encoder, lb = get_process_data
    ml_model = model.train_model(X_train=X, y_train=y)
    preds = ml_model.predict(X)
    precision, recall, fbeta = model.compute_model_metrics(y=y, preds=preds)
    assert isinstance(precision, float),  'precision'
    assert isinstance(recall, float), 'recall'
    assert isinstance(fbeta, float), 'f1'


def test_can_inference(get_process_data):
    X, y, encoder, lb = get_process_data
    ml_model = model.train_model(X_train=X, y_train=y)
    preds = model.inference(model=ml_model, X=X)
    assert isinstance(preds, np.ndarray)
