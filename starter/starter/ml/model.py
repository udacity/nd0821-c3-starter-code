import pickle
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

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

    dtc_model = DecisionTreeClassifier()
    dtc_model.fit(X_train, y_train)

    # Save model to file
    with open(os.path.join('starter', 'model', 'model_dtc.pkl'), 'wb') as file:
        pickle.dump(dtc_model, file)

    return dtc_model


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
    return model.predict(X)


def load_model(model_name):
    """Load a model from the model folder

    Args:
        model_name (pickle): name of the model to load

    Returns:
        model: sklearn model
    """
    return pickle.load(open(os.path.join('starter', 'model', model_name), 'rb'))


def load_encoder(encoder_name):
    return pickle.load(open(os.path.join('starter', 'model', encoder_name), 'rb'))


def load_lb(lb_name):
    return pickle.load(open(os.path.join('starter', 'model', lb_name), 'rb'))
