import pickle
import os
from starter.ml.data import process_data
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
    with open(os.path.join('model', 'model_dtc.pkl'), 'wb') as file:
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


def load_model(model_path):
    """Load a model from the model folder

    Args:
        model_path (pickle): name of the model to load

    Returns:
        model: sklearn model
    """
    return pickle.load(open(model_path, 'rb'))


def load_encoder(encoder_path):
    return pickle.load(open(encoder_path, 'rb'))


def load_lb(encoder_path):
    return pickle.load(open(encoder_path, 'rb'))


def performance_on_model_slices(test, cat_features, model, encoder, lb):
    """ computes and performance on model slices.

    Args:
        X: X data
        y: y data
        model: model for inference
        cat_features: category features
    """


    with open(os.path.join('outputs', 'slice_output.txt'),
              'w',
              encoding="utf-8") as file:

        for category in cat_features:
            cat_values = test[category].unique()
            file.write(f"Fixed Feature: {category}\n")

            for value in cat_values:
                # filter test_df
                filter_df = test[test[category] == value]
                if filter_df.shape[0] != 0:
                    # Proces the test data with the process_data function.
                    X_test, y_test, _, _ = process_data(
                        filter_df,
                        categorical_features=cat_features,
                        label="salary",
                        training=False,
                        encoder=encoder,
                        lb=lb,
                        )

                    preds = inference(model, X_test)
                    precision, recall, fbeta = compute_model_metrics(y_test, preds)
                    file.write(f"\t{category}={value}\t\tprecision:{round(precision,2)}\trecall{round(recall,2)}\tfbeta{round(fbeta,2)}\n")
