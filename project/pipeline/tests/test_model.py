import numpy as np
from project.pipeline.ml.model import compute_model_metrics, inference


def test_column_names(data):
    expected_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "salary"
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert all(item in these_columns for item in expected_columns), "Expected columns is present in dataset"


def test_inference(model, testprocess):
    pred = inference(model, testprocess[0])
    assert isinstance(pred, np.ndarray), "Prediction is NOT a numpy array, check your data"


def test_compute_model_metrics(model, testprocess):
    pred = inference(model, testprocess[0])
    precision, recall, fbeta = compute_model_metrics(testprocess[1], pred)
    assert isinstance(precision, float), "Precision is NOT float"
    assert isinstance(recall, float), "Recall is NOT float"
    assert isinstance(fbeta, float), "Fbeta is NOT float"
