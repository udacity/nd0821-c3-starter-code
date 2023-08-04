import os.path

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn import svm
import pickle


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
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf


def save_model(model, encoder, lb, model_path="./model"):
    """
    Saves a trained model and encoders using pickle to model_path
    Parameters
    ----------
    model:
        trained machine learning model
    encoder:
        trained OneHotEncoder
    lb:
        trained lb
    model_path: str
        Path to model file
    Returns
    -------

    """
    pickle.dump(model, open(os.path.join(model_path, "model.pkl"), "wb"))
    pickle.dump(encoder, open(os.path.join(model_path, "encoder.pkl"), "wb"))
    pickle.dump(lb, open(os.path.join(model_path, "lb.pkl"), "wb"))


def load_model(model_path="./model"):
    """
    Saves a trained model and encoders using pickle to model_path
    Parameters
    ----------
    model_path: str
        Path to model files
    Returns
    -------
    model:
        trained machine learning model
    encoder:
        trained OneHotEncoder
    lb:
        trained lb

    """
    try:
        model = pickle.load(open(os.path.join(model_path, "model.pkl"), "rb"))
        encoder = pickle.load(open(os.path.join(model_path, "encoder.pkl"), "rb"))
        lb = pickle.load(open(os.path.join(model_path, "lb.pkl"), "rb"))
    except FileNotFoundError as error:
        raise error
    except EOFError as error:
        raise error

    return model, encoder, lb


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
    """Run model inferences and return the predictions.
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
    output = model.predict(X)
    return output


def compute_model_performance_slice(df, feature, y_test, predictions):
    """
    Computes performance metrics on slices
    Parameters
    ----------
    df:
        test dataframe
    feature:
        Feature that is being sliced
    y_test:
        labels
    predictions:
        predicted labels

    Returns
    -------

    """
    slice_values = df[feature].unique().tolist()
    performance_df = pd.DataFrame(
        index=slice_values, columns=["precision", "recall", "fbeta"]
    )
    for value in slice_values:
        slice_mask = df[feature] == value

        slice_y = y_test[slice_mask]
        slice_predictions = predictions[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_predictions)

        performance_df.at[value, "precision"] = precision
        performance_df.at[value, "recall"] = recall
        performance_df.at[value, "fbeta"] = fbeta

    return performance_df
