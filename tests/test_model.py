#!/usr/bin/env -S python3 -i

"""
Testsuite for model checks.
author: I. Brinkmeier
date:   2023-09
"""

###################
# Imports
###################
import pytest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from src.config import get_config
from src.training.ml.data import process_data
from src.training.ml.model import train_model, inference, compute_model_metrics

###################
# Coding
###################

#
# arrange setup
#
# load in the configuration file
config_file = get_config()


#
# model tests
#

def test_model_evaluation(raw_test_data) -> None:
    """ Checks model type single xgb classifier evaluation metrics """
    assert len(raw_test_data) > 0, 'Empty dataframe for evaluation tests'

    processor, X, y = process_data(raw_test_data, label='salary', scaling=True, config_file=config_file)
    X_processed = processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        stratify=y_processed,
        test_size=config_file['model']['test_size'],
        random_state=config_file['model']['random_seed']
    )
    xgb_test = train_model(
                XGBClassifier(
                    objective=config_file['model']['xgb_cls']['objective'],
                    n_estimators=config_file['model']['xgb_cls']['n_estimators'],
                    eval_metric=config_file['model']['xgb_cls']['eval_metric'],
                    random_state=config_file['model']['random_seed'],
                    early_stopping_rounds=config_file['model']['early_stopping_rounds'],
                ),
                X_train, y_train, X_test, y_test,
                param_grid=None, config=config_file, test_run=True)

    assert isinstance(xgb_test, XGBClassifier), 'No XGBClassifier for evaluation tests'

    # check test predictions of single XGBClassifier instance
    y_preds = inference(xgb_test, X_test)
    precision, recall, fbeta, cm, cls_report = compute_model_metrics(y_test, y_preds)

    # using only a limited subset of data, metrics are not really good
    assert precision > 0.62, 'Test precision expected be above 0.62'
    assert recall > 0.47, 'Test recall expected be above 0.47'
    assert fbeta > 0.54, 'Test fbeta expected be above 0.54'
    assert cls_report['accuracy'] > 0.8, 'Test accuracy of classification report expected to be above 0.8'


if __name__ == "__main__":
    # as cli command to create the model test report:
    # pytest ./tests/test_model.py --html=./tests/model_test_report.html --capture=tee-sys
    pytest.main(["--html=model_test_report.html", "-v"])
