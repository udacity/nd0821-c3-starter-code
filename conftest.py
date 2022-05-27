import pytest
import os

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from starter.starter.ml.data import load_data


np.random.seed(42)


@pytest.fixture(scope='session')
def root_path():
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    return root_path


@pytest.fixture(scope='session')
def data():
    df = pd.DataFrame(np.random.randint(0, 400, size=(200, 4), ), columns=['A', 'B', 'C', 'D'])
    df['target'] = np.random.randint(0, 2, size=(200,))

    y = df['target']
    X = df.drop(['target'], axis=1)

    return X, y


@pytest.fixture(scope='session')
def train_data(root_path):
    df = load_data(root_path, "train_census.csv")

    zero_class = df[df['salary'] == '<=50K'].iloc[0].drop('salary')
    one_class = df[df['salary'] == '>50K'].iloc[0].drop('salary')

    return zero_class, one_class


@pytest.fixture(scope='session')
def model(data):
    X, y = data

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    return model