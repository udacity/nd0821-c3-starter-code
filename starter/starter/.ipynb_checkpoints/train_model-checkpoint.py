# Script to train machine learning model.
from joblib import dump, load
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import *
import pandas as pd

# Add the necessary imports for the starter code.
# Add code to load in the data.
def get_model(data_path: str,output_path: str):
    data=pd.read_csv(data_path)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test,encoder,lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder,lb=lb)
    # Train and save a model.
    lg=train_model(X_train,y_train)
    preds=inference(lg,X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    dump(lg,output_path)
    return precision, recall, fbeta 

def splice_testing(data_path: str, model_path: str, feature: str) -> None:
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    model=load(model_path)
    data=pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)
    with open('output.txt', 'w') as f:
        for val in test[feature].unique():
            df=test[test[feature] == val]
            X, y, encoder, lb = process_data(
            df, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb)
            preds=inference(model,X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            f.write(f"score for {val}: precision: {precision} recall: {recall} fbeta: {fbeta} \n")
