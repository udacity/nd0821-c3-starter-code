# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import joblib
import json

from ml.data import process_data
from ml.model import (evaluate_slices, train_model, inference,
                      compute_model_metrics)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "census.csv"

data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)

# Train a model.
model = train_model(X_train, y_train)

# Save model, encoder, and label binarizer.
MODEL_DIR = ROOT_DIR / "model"
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
joblib.dump(lb, MODEL_DIR / "label_binarizer.pkl")
joblib.dump(model, MODEL_DIR / "model.pkl")

# Evaluate overall model performance
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {fbeta:.3f}")

metrics_path = MODEL_DIR / "overall_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({"precision": precision, "recall": recall,
               "fbeta": fbeta}, f, indent=2)

# Evaluate categorical features performance. Used to check for subgroup bias
output_path = MODEL_DIR / "slice_output.txt"
evaluate_slices(data, cat_features, "salary", model, encoder, lb,
                output_path)
