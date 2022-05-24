import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(root_path, file_name):
    df = pd.read_csv(os.path.join(root_path, "data", file_name))

    return df

def save_data(df, root_path, file_name):
    df.to_csv(os.path.join(root_path, "data", file_name), index=False)

def process_data(
    data,
    categorical_features,
    label=None,
    training=True,
    preprocessor=None,
    label_binarizer=None,
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = data[label]
        X = data.drop([label], axis=1)
    else:
        y = np.array([])
        X = data

    continuous_features = list(set(X.columns) - set(categorical_features))

    if training is True:
        preprocessor = ColumnTransformer(
            transformers=[
                ("continuous_feats", StandardScaler(), continuous_features),
                ("categorical_feats", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features),
            ],
            remainder="drop",  # This drops the columns that we do not transform
        )

        label_binarizer = LabelBinarizer()

        X = preprocessor.fit_transform(X)
        y = label_binarizer.fit_transform(y.values).ravel()
    else:
        X = preprocessor.transform(X)

        # Catch the case where y is None because we're doing inference.
        try:
            y = label_binarizer.transform(y.values).ravel()
        except AttributeError:
            pass

    return X, y, preprocessor, label_binarizer


if __name__ == "__main__":
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    df = load_data(root_path, "clean_census.csv")

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

    X, y, preprocessor, label_binarizer = process_data(df, cat_features, 'salary')
