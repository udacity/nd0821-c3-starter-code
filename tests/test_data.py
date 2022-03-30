from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.ml.data import process_data


def test_raw_data(categorical_features, data):
    all_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    assert set(data.columns.values) == set(all_columns)
    assert set(data.columns.values).issuperset(set(categorical_features))


def test_process_data(categorical_features, data):

    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )

    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

    X, y, n_encoder, n_lb = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert n_encoder == encoder
    assert n_lb == lb
