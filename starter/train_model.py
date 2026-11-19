# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from ml.data import clean_data, process_data

# Add the necessary imports for the starter code.
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)
from ml.model_card import generate_model_card
from sklearn.model_selection import train_test_split

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

# Load the data from the project root.
print("Load and clean data")
data_path = Path(__file__).resolve().parents[1] / "data" / "census.csv"
raw_data = pd.read_csv(data_path, skipinitialspace=True)
data = clean_data(raw_data)
print(
    f"rows: {len(raw_data):,} -> {len(data):,}  (removed {len(raw_data) - len(data):,} duplicates/invalid)"  # noqa E501
)

metrics_path = (
    Path(__file__).resolve().parents[1] / "model" / "slice_output.txt"
)

# Optional: use K-fold cross validation instead of a train-test split.
print("Split data into train and test sets")
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
print("Training and saving model")
model = train_model(X_train, y_train)


# Model metrics
print("Generate metrics for model")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
overall = {"precision": precision, "recall": recall, "fbeta": fbeta}
print(pd.Series(overall).round(4).to_string())

# Slice metrics
print("Generate slice model metrics")
slice_frames = [
    compute_slice_metrics(
        test, feat, model, cat_features, encoder, lb, label="salary"
    )
    for feat in cat_features
]
slices = pd.concat(slice_frames, ignore_index=True)

with open(metrics_path, "w") as f:
    f.writelines(
        frame.round(4).to_string(index=False) + "\n\n"
        for frame in slice_frames
    )
print(f"Slice model metrics save to {metrics_path}")

print("generate model card")
generate_model_card(
    slices[slices.n >= 30].sort_values("fbeta").head(8).round(3),
    overall,
    30,
    data,
    len(raw_data),
    train,
    test,
    cat_features,
    data_path,
    Path(__file__).resolve().parents[1] / "MODEL_CARD.md",
    "salary",
)
