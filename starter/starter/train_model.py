# Script to train machine learning model.

# Add the necessary imports for the starter code.
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compute_model_metrics, inference


# Add code to load in the data.
path = os.path.join("starter", "data", "census.csv")
data = pd.read_csv(path)

# Remove leading spaces in columns
new_col = [c.lstrip() for c in data.columns.to_list()]
old_col = data.columns.to_list()

col_name_dict = {}
for o, n in zip(old_col, new_col):
    col_name_dict[o] = n

data = data.rename(col_name_dict, axis=1)

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
X_train, y_train, oh_encoder, lbl_binarizer = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=oh_encoder, lb=lbl_binarizer)

# Train and save a model.
rf_model = train_model(X_train, y_train)

# Get prediction using trained model
y_pred = inference(rf_model, X_test)

# Get model scores (precision, recall, fbeta)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)