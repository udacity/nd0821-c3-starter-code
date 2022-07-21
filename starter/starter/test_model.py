import pytest
import numpy as np
import pandas as pd
from ml.model import *
from ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression

@pytest.fixture(scope='session')
def data_():

    df = pd.read_csv('../../starter/data/census.csv')

    return df

@pytest.fixture(scope='session')
def split_data(data_):
    train, test = train_test_split(data_, test_size=0.20)
    
    return train, test

@pytest.fixture(scope='session')
def process_dataset(split_data):
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
    split_data[0], categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test,encoder,lb = process_data(
    split_data[1], categorical_features=cat_features, label="salary", training=False, encoder=encoder,lb=lb)
    
    return X_train, y_train,X_test, y_test

@pytest.fixture(scope='session')
def train_model_(process_dataset):
    lr=train_model(process_dataset[0],process_dataset[1])
    return lr

@pytest.fixture(scope='session')
def make_pred(process_dataset,train_model_):
    return inference(train_model_,process_dataset[2])

def test_train_model(process_dataset):
    lr = train_model(process_dataset[0],process_dataset[1])
    check_is_fitted(lr)

def test_inference(process_dataset,train_model_):
    preds = inference(train_model_,process_dataset[2])
    assert preds.shape[0]==process_dataset[3].shape[0]
def test_compute_model_metrics(process_dataset,make_pred):
    precision, recall, fbeta = compute_model_metrics(make_pred,process_dataset[3])
    assert type(precision)==np.float64
    assert type(recall)==np.float64
    assert type(fbeta)==np.float64