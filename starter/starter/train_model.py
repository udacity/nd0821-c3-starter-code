# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import *
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
data = pd.read_csv('data/census.csv')

# Add code to load in the data.

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
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')
