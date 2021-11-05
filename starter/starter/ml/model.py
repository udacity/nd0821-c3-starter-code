import logging
import numpy as np
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
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
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        # 'max_iter': [100, 200, 300, 400, 500, 1000],
    }
    logger.info('Training model with GridSearchCV: %s', param_grid)
    cv = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    cv.fit(X_train, y_train)
    logger.info('Best params: %s', cv.best_params_)
    logger.info('Best score: %s', cv.best_score_)
    return cv.best_estimator_


def compute_model_metrics(y: np.ndarray,
                          preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

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


def inference(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model.LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
