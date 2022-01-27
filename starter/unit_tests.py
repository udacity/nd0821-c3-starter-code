import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from starter.ml.data import process_data
from starter.ml.model import *
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


if __name__ == "__main__":
    test_train()
    test_inference()
    test_metrics()    
    print("Everything passed")