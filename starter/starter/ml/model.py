from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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
    rf_pipe = RandomForestClassifier()
    logger.info("Fitting")
    rf_pipe.fit(X_train,y_train)

    lr_pipe = LogisticRegression()
    lr_pipe.fit(X_train, y_train)

    return rf_pipe, lr_pipe
    

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

def slice_performance(model, data, encoder, lb, save_path):
    """ Outputs the performance of the model on slices of the data into a csv file.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    data : pd.Dataframe
        Data to be sliced and used for model prediction.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    save_path : String 
        Absolute path and filename to save the output )
    """
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    slice_performance = []

    for category in cat_features:
        unique_values = data[category].unique()
        for value in unique_values:
            slice_data = data[data[category] == value]
            X_slice, y_slice, encoder, lb = process_data(
                slice_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )
            y_slice_pred = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, y_slice_pred)
            slice_performance.append((category, value, precision, recall, fbeta))
            slice_performance.append('\n')

    with open(save_path, 'w', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(slice_performance)


    return 0
