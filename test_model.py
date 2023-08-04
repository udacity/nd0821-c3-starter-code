import os
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn import svm
from sklearn.exceptions import NotFittedError
from src.ml.model import train_model
from src.ml.model import save_model
from src.ml.model import load_model
from src.ml.model import compute_model_metrics
from src.ml.model import inference
from src.ml.model import compute_model_performance_slice
from src.ml.data import process_data
from src.config import cat_features


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv("./data/census.csv")


@pytest.fixture(scope="module")
def dataset_split(dataset):
    train, test = train_test_split(
        dataset,
        test_size=0.20,
    )
    return train, test


@pytest.fixture(scope="module")
def train_data(dataset_split):
    train, _ = dataset_split
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    return X_train, y_train, encoder, lb


@pytest.fixture(scope="module")
def test_data(dataset_split, train_data):
    _, test = dataset_split
    _, _, encoder, lb = train_data

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_test, y_test, test


@pytest.fixture(scope="module")
def model_data():
    model, encoder, lb = load_model()
    return model, encoder, lb


def test_train_model(train_data):
    X_train, y_train, encoder, lb = train_data
    model = train_model(X_train, y_train)

    try:
        model.predict(X_train)
    except NotFittedError as error:
        print(error)
        raise error


def test_save_model(model_data, tmp_path):
    model, encoder, lb = model_data
    save_model(
        model=model,
        encoder=encoder,
        lb=lb,
        model_path=str(tmp_path)
    )
    assert os.path.isfile(os.path.join(tmp_path, "model.pkl"))
    assert os.path.isfile(os.path.join(tmp_path, "encoder.pkl"))
    assert os.path.isfile(os.path.join(tmp_path, "lb.pkl"))


def test_load_model():
    model, encoder, lb = load_model()

    assert type(model) == type(svm.SVC())
    assert type(encoder) == type(OneHotEncoder())
    assert type(lb) == type(LabelBinarizer())


def test_inference(model_data, test_data):
    model, encoder, lb = model_data
    X_test, _, _ = test_data
    try:
        inference(model, X_test)
    except Exception as error:
        print(error)
        raise error


def test_compute_model_metrics(test_data):
    _, y_test, _ = test_data

    predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    try:
        precision, recal, fbeta = compute_model_metrics(y_test[:15], predictions)
    except Exception as error:
        print(error)
        raise error


def test_compute_model_performance_slice(test_data):
    _, y_test, test = test_data

    predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    try:
        precision, recal, fbeta = compute_model_performance_slice(test[:15], 'workclass', y_test[:15], predictions)
    except Exception as error:
        print(error)
        raise error
