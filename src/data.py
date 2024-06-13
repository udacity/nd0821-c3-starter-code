"""
Author: Eric Koch
Date Created: 2024-05-30

This modules provides code to load and process census data
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_raw_census_data():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__name__)), "data/census.csv"
    )
    return pd.read_csv(data_path)


def load_cleaned_census_data():
    data_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__name__)),
        "src/cleaning/clean_data.csv")
    return pd.read_csv(data_path)


def process_data(
        x_data,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        label_binarizer=None):
    """Process the data used in the machine learning pipeline.

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
        y_data = x_data[label]
        x_data = x_data.drop([label], axis=1)
    else:
        y_data = np.array([])

    x_categorical = x_data[categorical_features].values
    x_continuous = x_data.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        label_binarizer = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y_data = label_binarizer.fit_transform(y_data.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y_data = label_binarizer.transform(y_data.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    x_data = np.concatenate([x_continuous, x_categorical], axis=1)
    return x_data, y_data, encoder, label_binarizer
