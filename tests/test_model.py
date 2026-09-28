import pandas as pd
import numpy as np
from starter.ml import model
from starter.ml.data import process_data


def make_dummy_data():
    df = pd.DataFrame(
        {
            "age": [25, 40, 50, 22],
            "workclass": ["Private", "Self-emp", "Private", "Private"],
            "education": ["Bachelors", "HS-grad", "HS-grad", "Bachelors"],
            "marital-status": ["Never-married", "Married", "Married", "Never-married"],
            "occupation": ["Tech-support", "Exec-managerial", "Adm-clerical", "Sales"],
            "relationship": ["Not-in-family", "Husband", "Husband", "Own-child"],
            "race": ["White", "Black", "White", "White"],
            "sex": ["Male", "Female", "Female", "Male"],
            "capital-gain": [0, 1000, 0, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 50, 30, 20],
            "native-country": ["United-States", "United-States", "Canada", "United-States"],
            "salary": [0, 1, 0, 0],
        }
    )
    return df


def test_train_and_inference_roundtrip(tmp_path):
    df = make_dummy_data()
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
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    clf = model.train_model(X, y)
    preds = model.inference(clf, X)
    assert preds.shape[0] == y.shape[0]
    # predictions should be 0/1
    assert set(np.unique(preds)).issubset({0, 1})


def test_save_model(tmp_path):
    df = make_dummy_data()
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    clf = model.train_model(X, y)
    out = tmp_path / "out_model.joblib"
    path = model.save_model(clf, path=str(out))
    assert path == str(out)
    assert out.exists()


def test_evaluate_slices_returns_entries():
    df = make_dummy_data()
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    clf = model.train_model(X, y)
    results = model.evaluate_slices(clf, df, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb)
    # Should contain at least one slice key
    assert isinstance(results, dict)
    assert len(results) > 0
    # keys should look like feature=value
    assert any("=" in k for k in results.keys())
