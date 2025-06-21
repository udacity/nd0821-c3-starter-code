from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


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
    model = RandomForestClassifier(random_state=42)
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


def evaluate_slices(data, model, encoder, lb, cat_features, label_col="salary", slice_feature="marital-status"):
    """
    Evaluate model performance on slices of the data based on a categorical feature.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset to slice (e.g., test set).
    model : sklearn model
        Trained ML model.
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer from training.
    cat_features : list
        List of categorical features.
    label_col : str
        The name of the label column.
    slice_feature : str
        Categorical feature to slice on.
    """
    results = {}
    
    for value in data[slice_feature].unique():
        slice_data = data[data[slice_feature] == value]
        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=cat_features,
            label=label_col,
            training=False,
            encoder=encoder,
            lb=lb
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results[value] = (precision, recall, fbeta)
    return results
