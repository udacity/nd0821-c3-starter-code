"""
Author: Eric Koch
Date Created: 2024-05-30

This module scores our staged ("testing") model on our test data
"""
import logger as appLogger
from ml import model
from ml import data as datalib
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import os

logger = appLogger.logger

def run_all():
    try:
        # Load the data
        logger.info("Loading census data...")
        data = datalib.load_cleaned_census_data()
        logger.info("Census data loaded successfully.")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error(f"Error loading census data: {e}")
        return

    try:
        # Optional enhancement, use K-fold cross-validation instead of a train-test split
        logger.info("Splitting data into train and test sets...")
        _, test = train_test_split(data, test_size=0.20)
        logger.info("Data split into train and test sets successfully.")
    except ValueError as e:
        logger.error(f"Error splitting data: {e}")
        return
    
    rfc_model, encoder, lb = model.load_model(os.path.join(model.MODEL_FOLDER, model.TEST_MODEL_FILENAME))
    
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
        X_test, y_test, _, _ = datalib.process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        logger.info("Test data processed successfully.")
    except KeyError as e:
        logger.error(f"Error processing test data: {e}")
        return
    
    try:
        logger.info("Running model inference...")
        y_pred = model.inference(rfc_model, X_test)
        logger.info("Model inference completed successfully.")
    except NotFittedError as e:
        logger.error(f"Error during model inference: {e}")
        return

    try:
        logger.info("Computing model metrics...")
        precision, recall, fbeta = model.compute_model_metrics(y_test, y_pred)
        logger.info(f"Model metrics - Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")
    except ValueError as e:
        logger.error(f"Error computing model metrics: {e}")
    
    try:
        logger.info("Computing model performance on slices...")
        model.model_performance_on_slices(rfc_model, test, 'salary', cat_features, encoder, lb)
    except NotFittedError as e:
        logger.error(f"Error during model inference: {e}")
        return


if __name__ == "__main__":
    run_all()