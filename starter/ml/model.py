import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data

model_path = Path(__file__).resolve().parents[2] / "model" / "model.pkl"


def retrieve_model():
    if os.path.isfile(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)

    return None


def train_model(X_train, y_train, force=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    df, feature, model, cat_features, encoder, lb, label="salary"
):
    """Compute metrics for every unique value of `feature` in `df`.

    `df` must still contain the raw (unprocessed) columns and the label.
    Returns a DataFrame with one row per slice.
    """
    rows = []
    for value in sorted(df[feature].unique()):
        subset = df[df[feature] == value]
        X, y, _, _ = process_data(
            subset,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        precision, recall, fbeta = compute_model_metrics(
            y, inference(model, X)
        )
        rows.append((feature, value, len(subset), precision, recall, fbeta))
    return pd.DataFrame(
        rows, columns=["feature", "value", "n", "precision", "recall", "fbeta"]
    )
