from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
import logging
import pandas as pd
import numpy as np 
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def load_model(root_path, model_name):
    with open(os.path.join(root_path, "model", model_name), "rb") as f:
        model = pickle.load(f)

    return model

def save_model(model, root_path, model_name):
    with open(os.path.join(root_path, "model", model_name), "wb") as f:
        pickle.dump(model, f)

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
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    return model

def compute_model_metrics(y, preds):
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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def compute_slice_metrics(features,
                          labels,
                          predictions,
                          cat_features):

    """
    Computes the performance on categorical slices of the data
    Inputs:
        features: pandas DataFrame
            Contains the features on which a machine learning model was trained on
        labels: numpy array
            Ground-truth labels of each sample in features
        predictions: numpy array
            Predictions of each sample in features achieved by the model
        cat_features: list
            Categorical features on which we want to analyze the model performance
    Returns:
        slice_performance: pandas DataFrame
            Contains precision, recall, TNR, and NPV of the groups in cat_features
    """

    # Convert labels and predictions into pandas Series
    labels = pd.Series(np.squeeze(labels))
    predictions = pd.Series(np.squeeze(predictions))

    # Construct the full dataframe containing labels and predictions
    df = pd.concat([features, labels, predictions], axis=1)
    df.columns = list(features.columns) + ['labels', 'predictions']

    # Calculate TP, FP, TN, and FN
    TP = df[df['labels'] == 1].groupby(cat_features)['predictions'].sum()
    FP = df[df['labels'] == 1].groupby(cat_features)['predictions'].apply(lambda x: x.count() - x.sum())
    TN = df[df['labels'] == 0].groupby(cat_features)['predictions'].apply(lambda x: x.count() - x.sum())
    FN = df[df['labels'] == 0].groupby(cat_features)['predictions'].sum()

    precision = (TP / (TP + FP))
    recall = (TP / (TP + FN))
    TNR = (TN / (TN + FP))  # True Negative Rate
    NPV = (TN / (TN + FN))  # Negative Predictive Value
    f_score = 2*((precision * recall) / (precision + recall))

    slice_performance = pd.concat([precision, recall, TNR, NPV, f_score], axis=1)
    slice_performance.columns = ['Precision', 'Recall', 'TNR', 'NPV', 'F-Score']

    return slice_performance
