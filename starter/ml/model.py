from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def train_model(X_train, y_train):
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
    # Simple RandomForest classifier with sensible defaults
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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
    """ Run model inferences and return the predictions.

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
    preds = model.predict(X)
    return preds


def save_model(model, path="model/model.joblib"):
    """Save trained model to disk.

    Creates parent directory if needed.
    Returns the path saved to.
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    joblib.dump(model, path)
    return path


def evaluate_slices(model, data: pd.DataFrame, categorical_features, label, encoder=None, lb=None):
    """Evaluate model performance on slices of the data for each categorical feature.

    Returns a dict mapping "feature=value" -> (precision, recall, fbeta)
    """
    results = {}
    # If encoder/lb provided, use process_data from sibling module to transform
    from starter.ml.data import process_data

    # if encoder/lb not provided we need to fit them on the whole data
    if encoder is None or lb is None:
        from starter.ml.data import process_data

        _, _, encoder, lb = process_data(data, categorical_features=categorical_features, label=label, training=True)

    for cat in categorical_features:
        values = data[cat].dropna().unique()
        for val in values:
            slice_df = data[data[cat] == val]
            if slice_df.shape[0] == 0:
                continue
            X_slice, y_slice, _, _ = process_data(slice_df, categorical_features=categorical_features, label=label, training=False, encoder=encoder, lb=lb)
            if X_slice.shape[0] == 0:
                continue
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            results[f"{cat}={val}"] = (precision, recall, fbeta)
    return results
