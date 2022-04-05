"""
Script for training a machine learning model

"""
# Script to train machine learning model.
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from ml import model
from ml.data import process_data

# Add code to load in the data.
data = pd.read_csv(config.DATA_PATH)


# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=config.TEST_SPLIT_SIZE)
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=config.cat_features, label=config.TARGET, training=True
)
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=config.cat_features, label=config.TARGET, training=False, encoder=encoder, lb=lb
)


# Train and save a model.
mod = model.train_model(X_train, y_train)

# Dump the model as a pickle file
with open(config.MODEL_PATH, "wb") as file:
    pkl.dump([encoder, lb, mod], file)

# Inference and evaluation
train_pred = model.inference(mod, X_train)
test_pred = model.inference(mod, X_test)
precision, recall, f_one = model.compute_model_metrics(y_test, test_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f_one}")

model.compute_metrics_by_slice(
    df=test,
    model=mod,
    encoder=encoder,
    lb=lb,
    cat_columns=config.cat_features,
    target=config.TARGET,
    output_path=config.METRICS_OUTPUT_PATH
)
