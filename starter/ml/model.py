import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


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
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


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
    return model.predict(X)


def save_artifact(artifact, path):
    """ Save a model or fitted preprocessor to `path` with joblib. """
    joblib.dump(artifact, path, compress=3)


def load_artifact(path):
    """ Load a model or fitted preprocessor saved with `save_artifact`. """
    return joblib.load(path)


def compute_slice_metrics(model, data, feature, categorical_features, label, encoder, lb):
    """ Compute the model metrics for each unique value of a categorical feature.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Dataframe containing the features and label.
    feature : str
        Name of the categorical feature to slice on.
    categorical_features : list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer.
    Returns
    -------
    slices : pd.DataFrame
        One row per value of `feature` with columns feature, value, n, precision,
        recall and fbeta.
    """
    rows = []
    for value in sorted(data[feature].unique()):
        slice_df = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        rows.append({
            "feature": feature,
            "value": value,
            "n": len(slice_df),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        })
    return pd.DataFrame(rows)
