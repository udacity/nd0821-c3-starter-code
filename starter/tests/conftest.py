import pytest
import pandas as pd
from starlette.testclient import TestClient
from starter.starter.ml.data import process_data
from starter.main import app


@pytest.fixture(scope="module")
def clean_data_df():
    return pd.read_csv('./starter/data/census_clean_test.csv', nrows=100)


@pytest.fixture(scope="module")
def get_process_data(clean_data_df):
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    label = 'salary'
    X, y, encoder, lb = process_data(X=clean_data_df,
                                     categorical_features=categorical_features,
                                     label=label,
                                     training=True,
                                     encoder=None,
                                     lb=None
                                     )
    return X, y, encoder, lb


@pytest.fixture(scope="module")
def test_app():
    # set up
    with TestClient(app) as test_client:
        # testing
        yield test_client

    # tear down