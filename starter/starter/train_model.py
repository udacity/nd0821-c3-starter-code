# Script to train machine learning model.

import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Add the necessary imports for the starter code.
from ml.data import make_dataset

# Add code to load in the data.
# Fist we generate the processed dataset.csv from the raw census.csv
make_dataset(os.path.join('starter', 'data', 'census.csv'),
             os.path.join('starter', 'data', 'dataset.csv'))

# Then, we load the processed datset
data = pd.read_csv(os.path.join('starter', 'data', 'dataset.csv'))
                   
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

# Train and save a model.
