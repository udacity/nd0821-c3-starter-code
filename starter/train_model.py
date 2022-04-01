from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
from joblib import dump, load
import os
import logging

from metrics import calculate_slice_metrics

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

data = pd.read_csv("../data/census_clean.csv")
logger.info("Splitting data ...")
train, test = train_test_split(data, test_size=0.20)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
logger.debug(f"Total columns in 'train': {len(train.columns)}")
logger.debug(f"Columns in 'train': {train.columns.values}")
logger.debug(f"Total columns in 'test': {len(test.columns)}")
logger.info("Preprocessing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_features, label="salary", training=True
)
logger.info("Preprocessing train data is done")
logger.debug(f"X_train shape: {X_train.shape}")
logger.info("Preprocessing test data")
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=categorical_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
logger.info("Preprocessing test data is done")
logger.debug(f"X_test shape: {X_train.shape}")
logger.info("Training the model")
model = train_model(X_train, y_train)
logger.info("Training model is done")
MODEL_PATH = "../model"
MODEL_NAME = "cl_model.joblib"
ENCODER_NAME = "encoder.joblib"
LB_NAME = "lb.joblib"
logger.info("Saving models")
dump(model, os.path.join(MODEL_PATH, MODEL_NAME))
dump(encoder, os.path.join(MODEL_PATH, ENCODER_NAME))
dump(lb, os.path.join(MODEL_PATH, LB_NAME))
logger.info("Models are saved")
loaded_model = load(os.path.join(MODEL_PATH, MODEL_NAME))
preds = inference(loaded_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
calculate_slice_metrics(
    model=loaded_model,
    encoder=encoder,
    lb=lb,
    categorical_features=categorical_features,
    test_data=test,
)
