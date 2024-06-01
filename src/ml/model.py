"""
Author: Eric Koch
Date Created: 2024-05-30

This module provides code to train and score the model.
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import ml.data as datalib

# Define constants
def find_repo_root(current_path):
    """
    Finds the root of the repository by looking for the .git directory.

    Parameters
    ----------
    current_path : str
        The current file path.

    Returns
    -------
    root_path : str
        The root path of the repository.
    """
    while current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError("Repository root with .git directory not found.")

REPO_ROOT = find_repo_root(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(REPO_ROOT, "model")
TEST_MODEL_FILENAME = "test_model.pkl"
PROD_MODEL_FILENAME = "prod_model.pkl"

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.

    Returns
    -------
    model : object
        Trained machine learning model.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('clf', RandomForestClassifier(random_state=42))  # Classifier
    ])
    
    param_grid = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def save_model(model, encoder, lb, filename):
    """
    Saves the trained machine learning model, encoder, and label binarizer to a file.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    encoder : object
        Encoder used for transforming categorical features.
    lb : object
        Label binarizer used for transforming labels.
    filename : str
        Path to the file where the model should be saved.
    """
    model_path = os.path.join(MODEL_FOLDER, filename)
    joblib.dump({'model': model, 'encoder': encoder, 'lb': lb}, model_path)
    print(f"Model saved to {model_path}")

def load_model(filename):
    """
    Loads a machine learning model, encoder, and label binarizer from a file.

    Parameters
    ----------
    filename : str
        Path to the file from which the model should be loaded.

    Returns
    -------
    model : object
        Loaded machine learning model.
    encoder : object
        Loaded encoder.
    lb : object
        Loaded label binarizer.
    """
    model_path = os.path.join(MODEL_FOLDER, filename)
    data = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return data['model'], data['encoder'], data['lb']

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1 score.

    Parameters
    ----------
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
    """
    Run model inferences and return the predictions.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    encoder : object
        Encoder used for transforming categorical features.
    lb : object
        Label binarizer used for transforming labels.

    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # We expect X to be passed in encoded
    # X_encoded = encoder.transform(X)
    preds = model.predict(X)
    return preds

def model_performance_on_slices(model, data, target_column, categorical_features, encoder, lb):
    """
    Outputs the performance of the model on slices of the data based on categorical features.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    data : pd.DataFrame
        DataFrame containing the data used for prediction and the target column.
    target_column : str
        The name of the target column in the DataFrame.
    categorical_features : list of str
        List of categorical feature names.
    encoder : object
        Encoder used for transforming categorical features.
    lb : object
        Label binarizer used for transforming labels.

    Returns
    -------
    None
    """
    # Separate the features and the target variable
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Transform the target variable using the label binarizer
    y = lb.transform(y)

    for feature in categorical_features:
        unique_values = data[feature].unique()
        for value in unique_values:
            slice_mask = (data[feature] == value)
            X_slice = X[slice_mask]
            y_slice = y[slice_mask]

            if X_slice.shape[0] == 0:
                continue

            # Process the slice of data
            X_slice_processed, _, _, _ = datalib.process_data(
                X_slice, categorical_features=categorical_features, label=None, training=False, encoder=encoder, lb=lb
            )

            preds = model.predict(X_slice_processed)

            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            print(f"Performance for {feature} = {value}:")
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            print(f"  F-beta: {fbeta}\n")