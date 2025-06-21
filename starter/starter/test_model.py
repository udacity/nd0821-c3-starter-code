import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    # Dummy data
    X_train = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y_train = np.array([0, 1, 0, 1])
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_inference():
    X_train = np.array([[0, 1], [1, 0]])
    y_train = np.array([0, 1])
    X_test = np.array([[0, 1], [1, 0]])

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert all(p in [0, 1] for p in preds)
