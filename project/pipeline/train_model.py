# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

# Add code to load in the data.
data = pd.read_csv("../data/clean_census.csv")
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
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
print("model OK")

# Save encoder and lb
joblib.dump(encoder, r"../model/encoder.pkl")
joblib.dump(lb, r"../model/lb.pkl")

# Inference
preds = inference(model, X_test)

# Model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")

