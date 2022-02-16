'''
Author: Oliver
Date: February 2022
'''
# Script to train machine learning model.
from matplotlib.font_manager import json_dump, json_load
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

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

num_features = [
    "age",
    "fnlgt",
    "education-num",
    "capital-gain",
    "capital-loss",  
    "hours-per-week",
]


def save_model_artifacts(file_dir, name, model, encoder, score):

    # save the model and the OneHot encoder
    model_pth = str(file_dir + '/' + name + '_model.pkl')
    encoder_pth = str(file_dir + '/' + name + '_encoder.pkl')
    score_pth = str(file_dir + '/' + name + '_score.json')

    if os.path.exists(file_dir):
        joblib.dump(model, model_pth)
        joblib.dump(encoder, encoder_pth)
        json_dump(score, score_pth)
        logger.info(f"Model and encoder saved to: {file_dir}")
    else:
        logger.error("Failed to save the model and encoder. Filepath incorrect!")


def load_model_artifacts(file_dir, name):

    # load model and encoder
    model_pth = str(file_dir + '/' + name + '_model.pkl')
    encoder_pth = str(file_dir + '/' + name + '_encoder.pkl')
    score_pth = str(file_dir + '/' + name + '_score.json')

    if os.path.exists(file_dir):
        model = joblib.load(model_pth)
        encoder = joblib.load(encoder_pth)
        score = json_load(score_pth)
        logger.info(f"Model and encoder loaded from directory {file_dir} ({name})")
    else:
        logger.error("Failed to load the model or encoder. Filepath incorrect!")

    return model, encoder, score


def start_train_pipeline(file_pth):

    # Add code to load in the data.
    df = pd.read_csv(file_pth)
    logger.info(f"File read, training process started: {file_pth}")
    
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(df, test_size=0.20)
    logger.info(f"Data is split into train and test.")

    # Proces the train/test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logger.info(f"Train {X_train.shape} and test {X_test.shape} data preprocessed with label salary.")

    # Train and save a model.
    model = train_model(X_train, y_train)
    logger.info(f"Model trained.")

    test_preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
    score = {"precision": precision, "recall": recall, "fbeta": fbeta}
    logger.info(f"Metrics: {score}")
    
    return model, encoder, score

if __name__ == "__main__":
    try:

        model, encoder, score = start_train_pipeline("./data/cleaned_census.csv")
        save_model_artifacts(str("./model"), str("lr"), model, encoder, score)
        # model, encoder, score = load_model_artifacts(str("./model"), str("lr"))
        # inference(model, X)

    except (Exception) as error:
        print("main error: %s", error)