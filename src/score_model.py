"""
Author: Eric Koch
Date Created: 2024-05-30

This module scores our staged ("testing") model on our test data
"""

import os

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

import logger as appLogger
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
        _, test = train_test_split(data, test_size=0.20)
        logger.info("Data split into train and test sets successfully.")
    except ValueError as err:
        logger.error(f"Error splitting data: {err}")
        return

    rfc_model, encoder, lb = model.load_model(
        os.path.join(model.MODEL_FOLDER, model.TEST_MODEL_FILENAME)
    )

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
        # Test the model against our test data
        logger.info("Processing test data...")
        x_test, y_test, _, _ = datalib.process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        logger.info("Test data processed successfully.")
    except KeyError as err:
        logger.error(f"Error processing test data: {err}")
        return

    try:
        logger.info("Running model inference...")
        y_pred = model.inference(rfc_model, x_test)
        logger.info("Model inference completed successfully.")
    except NotFittedError as err:
        logger.error(f"Error during model inference: {err}")
        return

    try:
        logger.info("Computing model metrics...")
        precision, recall, fbeta = model.compute_model_metrics(y_test, y_pred)
        logger.info(
            f"Model metrics - Precision: {precision}, Recall: {recall}, F-beta: {fbeta}"
        )
    except ValueError as err:
        logger.error(f"Error computing model metrics: {err}")

    try:
        logger.info("Computing model performance on slices...")
        performance = model.model_performance_on_slices(
            rfc_model, test, "salary", cat_features, encoder, lb
        )
        for slice_key, metrics in performance.items():
            print(f"Performance for {slice_key}:")
            print(f"  Precision: {metrics['precision']}")
            print(f"  Recall: {metrics['recall']}")
            print(f"  F-beta: {metrics['fbeta']}\n")
    except NotFittedError as err:
        logger.error(f"Error during model inference: {err}")
        return


if __name__ == "__main__":
    run_all()
