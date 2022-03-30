import pytest
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data


@pytest.fixture
def categorical_features() -> [int]:
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.read_csv("data/census_test.csv")


@pytest.fixture
def model():
    model = load("model/cl_model.joblib")
    return model

@pytest.fixture
def data_train(categorical_features, data):
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    return _load_process_data(categorical_features, train)


@pytest.fixture
def data_test(categorical_features, data):
    _, test = train_test_split(data, test_size=0.20, random_state=42)
    return _load_process_data(categorical_features, test)


@pytest.fixture
def predictions(categorical_features, data):
    _, test = train_test_split(data, test_size=0.20, random_state=42)
    X_test, _ = _load_process_data(categorical_features, test)
    model = load("model/cl_model.joblib")
    return model.predict(X_test)


def _load_process_data(categorical_features, data):
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")
    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_test, y_test
