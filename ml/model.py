from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from typing import List

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = LogisticRegression(
        max_iter=1000,
        random_state=23,
    )

    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
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
    model : sklearn.linear_model._logistic.LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_slices(data, cat_features, label, model, encoder, lb,
                    output_path) -> None:
    """Output to a file the performance on slices of just the categorical
    features
    """
    results: List[str] = []
    for feature in cat_features:
        for category in data[feature].unique():
            subset = data[data[feature] == category]
            X_slice, y_slice, _, _ = process_data(
                subset,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds_slice = inference(model, X_slice)
            p, r, f = compute_model_metrics(y_slice, preds_slice)
            results.append(
                f"Feature: {feature} | Category: {category} | "
                f"Precision: {p:.3f} | Recall: {r:.3f} "
                f"Fbeta: {f:.3f} | Number of samples: {len(y_slice)}"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(results))
