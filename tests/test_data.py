from starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def test_raw_data(data):
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
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    assert set(data.columns.values) == set(all_columns)
    assert set(data.columns.values).issuperset(set(categorical_features))


def test_process_data(data):

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

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
    assert n_lb == n_lb
