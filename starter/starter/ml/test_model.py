'''
Unit test for function in model.py
'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from data import process_data
from model import train_model, inference, compute_model_metrics

def test_train_model(X_train, y_train):
    model = train_model(X_train, y_train)
    assert type(model) == LogisticRegression

def test_compute_model_metrics(y, preds):
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64

def test_inference(model, x):
    preds = inference(model, x)
    assert type(preds) == np.ndarray

if __name__ == "__main__":
    data = pd.read_csv('../../data/census.csv')

    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _= process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    test_train_model(X_train, y_train)
    model = train_model(X_train, y_train)

    test_inference(model, X_test)
    y_pred_test = inference(model, X_test)

    test_compute_model_metrics(y_test, y_pred_test)
