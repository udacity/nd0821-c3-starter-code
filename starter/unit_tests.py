import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from starter.ml.data import process_data
from starter.ml.model import *
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import *
import joblib


class Base():
    def __init__(self):
        data = pd.read_csv('data/census.csv')
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
        X, y, encoder, lb = process_data(
            data, categorical_features=cat_features, label="salary", training=True
        )
        model = joblib.load("model/model.pkl")
        self.X = X
        self.y = y
        self.model = model
        
base = Base()

def test_train():
    X, y = base.X, base.y
    model = train_model(X, y)
    assert len(model.get_params()) > 0, "Model fitting failed"

def test_inference():
    preds = inference(base.model, base.X)
    assert np.all(preds >= 0) & np.all(preds <= 1), "Inference failed"

def test_metrics():
    preds = inference(base.model, base.X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert all([(x >=0) & (x <= 1) for x in [precision, recall, fbeta]]), "Metricss failed"    

def test_get():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200, "Get failed"
    assert 'greeting' in response.json(), "Get message not found"
    
def test_post_pred0():
    client = TestClient(app)
    response = client.post(
        "/",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        },
    )
    assert response.status_code == 200, "Post failed"
    assert response.json()['prediction'] == 0, "Unexpected prediction"
    
def test_post_pred1():
    client = TestClient(app)
    response = client.post(
        "/",
        json={
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        },
    )
    assert response.status_code == 200, "Post failed"
    assert response.json()['prediction'] == 1, "Unexpected prediction"




if __name__ == "__main__":
    test_train()
    test_inference()
    test_metrics()    
    test_get()
    test_post_pred0()
    test_post_pred1()
    print("Everything passed")