from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.starter.ml.data import process_data


def test_process_data(clean_data_df):
    features = clean_data_df
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    label = 'salary'
    X, y, encoder, lb = process_data(features, categorical_features, label, training=True, encoder=None, lb=None)

    assert clean_data_df.shape == (100, 15)
    assert X.shape == (100, 65)
    assert y.shape == (100,)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
