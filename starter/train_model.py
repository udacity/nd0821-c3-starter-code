# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data

# Add the necessary imports for the starter code.

# Load the data from the project root.
data_path = Path(__file__).resolve().parents[1] / "data" / "census.csv"
data = pd.read_csv(data_path, skipinitialspace=True)

# Optional: use K-fold cross validation instead of a train-test split.
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

# Process the test data with the process_data function.

# Train and save a model.
