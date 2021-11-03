import pandas as pd
import pytest
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.starter.ml.data import process_data


@pytest.fixture()
def clean_data_df():
    return pd.read_csv('./starter/data/census_clean.csv')


def test_process_data(clean_data_df):
    features = clean_data_df
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    label = 'salary'
    X, y, encoder, lb = process_data(features, categorical_features, label, training=True, encoder=None, lb=None)

    assert clean_data_df.shape == (32561, 15)
    assert X.shape == (32561, 108)
    assert y.shape == (32561,)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
