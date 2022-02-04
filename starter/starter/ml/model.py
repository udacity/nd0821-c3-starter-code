import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.data import process_data


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
    model = RandomForestClassifier()
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


def performance_on_slices(data, encoder, lb, cat_features, model):
    """ Get performance metrics for each categorical value slice.

    Inputs
    ------
    data : data for inference  
    encoder : Trained sklearn OneHotEncoder
    lb : Trained sklearn LabelBinarizer
    cat_features : Categorical features.
    model : Trained machine learning model.
    Returns
    -------
    result_df : pandas dataframe with performance metrics per categorical value slice
    """
    results_list = list()
    # Proces the input data with the process_data function.
    X, y, _, _ = process_data(
        data, 
        categorical_features=cat_features,
        encoder = encoder,
        lb = lb,
        label="salary",        
        training=False
    )    
    for cat_column in cat_features:
        cat_values = list(data[cat_column].unique())
        for cat_value in cat_values:
            # Getting index of categorical value
            cat_value_index = data[data[cat_column] == cat_value].index.values
            # Run inference and get performance on data subset
            preds_subset = inference(model, X[cat_value_index])
            precision, recall, fbeta = compute_model_metrics(y[cat_value_index], preds_subset)
            results_list.append([cat_column, cat_value, precision, recall, fbeta])
    result_df = pd.DataFrame(results_list, 
                             columns = ['cat_column', 'cat_value', 'precision', 'recall', 'fbeta'])
    return result_df