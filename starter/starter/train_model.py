# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.model import evaluate_slices

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_preds = inference(model, X_test)

# calculate the performance metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {fbeta:.4f}")

# build the model on slices of the data
slice_metrics = evaluate_slices(
    data=test,
    model=model,
    encoder=encoder,
    lb=lb,
    cat_features=cat_features,
    label_col="salary",
    slice_feature="marital-status"
)

# print out the performance metrics 
for status, (precision, recall, f1) in slice_metrics.items():
    print(f"marital-status = {status}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}\n")

