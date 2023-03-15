import pytest
import joblib
from sklearn.model_selection import train_test_split
from project.pipeline.ml.data import process_data, import_data


@pytest.fixture(scope='session')
def data():
    try:
        data = import_data("../../data/clean_census.csv")
        return data
    except FileNotFoundError as err:
        pytest.fail("Testing import data: File wasn't found")

@pytest.fixture(scope='session')
def model():
    try:
        model = joblib.load("../../model/rfc_model.pkl")
        return model
    except FileNotFoundError as err:
        pytest.fail("Testing import model: File wasn't found")

@pytest.fixture(scope='session')
def lb():
    try:
        model = joblib.load("../../model/lb.pkl")
        return model
    except FileNotFoundError as err:
        pytest.fail("Testing import lb: File wasn't found")

@pytest.fixture(scope='session')
def encoder():
    try:
        model = joblib.load("../../model/encoder.pkl")
        return model
    except FileNotFoundError as err:
        pytest.fail("Testing import encoder: File wasn't found")

@pytest.fixture(scope='session')
def testprocess(data, encoder, lb):
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
    try:
        train, test = train_test_split(data, test_size=0.20)
        X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary",
                                                     training=True)
        X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False,
                                            encoder=encoder, lb=lb)
        return [X_test, y_test]
    except AssertionError as err:
        raise err
