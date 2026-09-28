# Script to train machine learning model.
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_artifact,
    train_model,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
SLICE_OUTPUT_PATH = ROOT / "slice_output.txt"

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


def main():
    data = pd.read_csv(DATA_PATH)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
    print(f"Test precision: {precision:.4f}, recall: {recall:.4f}, F1: {fbeta:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    save_artifact(model, MODEL_DIR / "model.joblib")
    save_artifact(encoder, MODEL_DIR / "encoder.joblib")
    save_artifact(lb, MODEL_DIR / "lb.joblib")

    slices = pd.concat(
        [
            compute_slice_metrics(model, test, feature, cat_features, "salary", encoder, lb)
            for feature in cat_features
        ],
        ignore_index=True,
    )
    slice_report = slices.to_string(index=False, float_format="%.4f")
    print(slice_report)
    SLICE_OUTPUT_PATH.write_text(slice_report + "\n")
    print(f"Slice metrics written to {SLICE_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
