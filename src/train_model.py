"""
Author: Eric Koch
Date Created: 2024-05-30

This module integrates the data/model libraries to run the end-to-end training.
"""

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

import src.logger as appLogger
from src import data as datalib
from src import model

logger = appLogger.logger


def run_all():
    try:
        # Load the data
        logger.info("Loading census data...")
        data = datalib.load_cleaned_census_data()
        logger.info("Census data loaded successfully.")
    except (FileNotFoundError, pd.errors.EmptyDataError) as err:
        logger.error(f"Error loading census data: {err}")
        return

    try:
        # Optional enhancement, use K-fold cross-validation instead of a
        # train-test split
        logger.info("Splitting data into train and test sets...")
        train, test = train_test_split(data, test_size=0.20)
        logger.info("Data split into train and test sets successfully.")
    except ValueError as err:
        logger.error(f"Error splitting data: {err}")
        return

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

    try:
        logger.info("Processing training data...")
        x_train, y_train, encoder, lb = datalib.process_data(
            train, categorical_features=cat_features, label="salary", training=True)
        logger.info("Training data processed successfully.")
    except KeyError as err:
        logger.error(f"Error processing training data: {err}")
        return

    try:
        # Train and save the model
        logger.info("Training model...")
        rfc_model = model.train_model(x_train, y_train)
        logger.info("Model trained successfully.")
        logger.info("Saving model...")
        model.save_model(rfc_model, encoder, lb, model.TEST_MODEL_FILENAME)
        logger.info("Model saved successfully.")
    except (ValueError, NotFittedError) as err:
        logger.error(f"Error training or saving model: {err}")
        return


if __name__ == "__main__":
    run_all()
