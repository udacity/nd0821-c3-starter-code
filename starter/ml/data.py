import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def remove_processed_dataset(processed_dataset_name: str):
    os.remove(os.path.join('data', processed_dataset_name))


def make_dataset(raw_dataset_name: str, processed_dataset_name: str) -> pd.DataFrame:
    """Creates processed dataset from raw dataset ('census.csv') and save it
    to starter/data folder.

    Inputs
    ------
    raw_dataset_name: str
        relative path with name of the raw dataset ('census.csv')

    processed_dataset_name: str
        relative path with name of the dataset

    Returns
    -------
    dataset : pd.Dataframe
        project processed dataset
    """
    raw_dataset_path = os.path.join('data', raw_dataset_name)
    processed_dataset_path = os.path.join('data', processed_dataset_name)
    df_raw = pd.read_csv(
        raw_dataset_path, sep=",", skipinitialspace=True, na_values="?"
    )

    # make columns compatible with FASTAPI BaseModel
    df_raw.columns = df_raw.columns.str.replace('-', '_')

    # save datset to processed folder
    df_raw.to_csv(processed_dataset_path, index=False)

    return df_raw


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`

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
        data.

    y : np.array
        labels if labeled=True, otherwise empty np.array.

    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.

    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()

        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
