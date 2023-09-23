#!/usr/bin/env -S python3 -i

"""
Script to train machine learning model for binary classification.
author: I. Brinkmeier
date:   2023-09
"""

#####################
# Imports
#####################
from config import get_config, get_data_path, get_models_path
from ml.data import load_data, process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    plot_logloss_diagram,
    plot_error_diagram,
    plot_roc_curve_diagram
)

from datetime import datetime
TODAY = datetime.today().strftime('%Y-%m-%d_%H-%M')

import os
import sys
MAIN_DIR = os.path.join(os.getcwd(), 'src/')
sys.path.append(MAIN_DIR)
sys.path.append(os.getcwd())
print(f'-> train_model: sys.path: {sys.path}')

import logging
import logging.config
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#####################
# Coding
#####################

def go():
    ''' Checks the training workflow for census data with configuration parameters.

    Note:
    If as starting point no data could be read in, system exit happens.

    By now, XGBoost binary classifier is used with and without GridSearchCV.
    Regarding Hyperparameter:
    Out-of-scope is usage of Hyperopt, which is a Python library for optimizing both
    discrete and continuous hyperparameters for XGBoost instances.
    Hyperopt uses Bayesian optimization to tune hyperparameters, so, it would return
    a better prediction because it is not limited to the explicit given values of a
    parameter grid as defined for GridSearchCV.

    Regarding Logging:
    For training 'staging' logger is used.
    '''

    # load in the configuration file
    config_file = get_config()

    # get logging properties
    # info see: https://realpython.com/python-logging-source-code/
    logging.config.dictConfig(config_file['logging'])
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger('staging')
    
    # load in the data
    data = os.path.join(get_data_path(), config_file['etl']['orig_census'])
    logger.info("Load data from path: %s", data)
    try:
        df_data = load_data(data)
        logger.info("Census dataset: %s samples with %s features each, columns are: %s",
                    df_data.shape[0], df_data.shape[1], df_data.columns)
        logger.info('df_data head() of first 5 rows:\n %s', df_data.head())
    except Exception as e:
        logger.exception("Exit: exception of type %s occurred. Details: %s", type(e).__name__, str(e))
        sys.exit(1)

    # apply preprocessing,
    # for selected classifiers (like RandomForestClassifier, XGBoostClassifier) scaling is needed,
    # for stratification we need the target label column
    logger.info('Process data with preprocessing and drop target feature "salary"')
    processor, X, y = process_data(df_data,
                                   label='salary',
                                   scaling=True,
                                   config_file=config_file)

    X_processed = processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    # save ColumnTransformer processor in model dir
    logger.info("Save ColumnTransformer as final pickle file in models dir")
    col_transformer_artifact_label = config_file['model']['col_transformer']
    filename = os.path.join(get_models_path(), col_transformer_artifact_label)
    with open(filename, 'wb') as f:                                   
        joblib.dump(processor, f)

    # train test split
    # make sure splits contain same amount of categories in both sets via stratify param
    logger.info('Create train-test split with processed y target for stratifying')
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        stratify=y_processed,
        test_size=config_file['model']['test_size'],
        random_state=config_file['model']['random_seed']
    )

    # XGBoost Classifier
    # https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390
    # note: “binary:logistic” – logistic regression for binary classification, output probability
    logger.info('--- Start binary classification workflow for XGBClassifier ... ---')

    # first workflow:
    # single prediction with default classifier hyperparameters
    xgb_cls = train_model(
                xgb.XGBClassifier(
                    objective=config_file['model']['xgb_cls']['objective'],
                    n_estimators=config_file['model']['xgb_cls']['n_estimators'],
                    eval_metric=config_file['model']['xgb_cls']['eval_metric'],
                    random_state=config_file['model']['random_seed'],
                    early_stopping_rounds=config_file['model']['early_stopping_rounds'],
                ),
                X_train, y_train, X_test, y_test,
                param_grid=None,
                config=config_file, test_run=None)
    logger.info("Single XGBClassifier model training params:\n %s", xgb_cls)

    # plot and store evaluation results diagram
    # plot logloss diagramm
    title = 'Single XGB Classifier: Resulting Loss Diagram'
    # plot function adds date as first part of png name
    png_name = '_xgb-single_logloss_treeNo_diagram.png'
    plot_logloss_diagram(xgb_cls, config_file, title, png_name)

    # plot classification error
    title = 'Single XGB Classifier: Resulting Classification Error Diagram'
    png_name = '_xgb-single_class_error_diagram.png'
    plot_error_diagram(xgb_cls, config_file, title, png_name)

    # plot ROC curve diagram
    # https://www.turing.com/kb/auc-roc-curves-and-their-usage-for-classification-in-python
    title = 'Single XGB Classifier: Resulting ROC Curve Diagram'
    png_name = '_xgb-single_roc-curve_diagram.png'
    plot_roc_curve_diagram(xgb_cls,
                           X_train, y_train,
                           X_test, y_test,
                           config_file, title, png_name)

    # prediction with created test data from split and get metrics on it
    y_preds = inference(xgb_cls, X_test)
    precision, recall, fbeta, cm, cls_report = compute_model_metrics(y_test, y_preds)
    print('Validation:')
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'fbeta : {fbeta}')
    logger.info("--- Validation metrics of Single XGBClassifier: ---")
    logger.info("Precision: %s, Recall: %s, Fbeta: %s", precision, recall, fbeta)
    print('Confusion Matrix')
    print(cm)
    print()
    print('Classification Report')
    print(cls_report)
    print('---------')
    logger.info('- Confusion Matrix -')
    logger.info(cm)
    logger.info('- Classification Report -')
    logger.info(cls_report)
    logger.info("----------")

    # save basic XGBoost model in model dir
    logger.info("Save single, basic XGBClassifier as pickle file in models dir")
    artifact_label = ''.join([TODAY, '_', config_file['model']['xgb_cls']['output_artifact']])
    filename = os.path.join(get_models_path(), artifact_label)
    with open(filename, 'wb') as f:                                     
        joblib.dump(xgb_cls, f)

    # second workflow:
    # usage of GridSearchCV (cross validataion approach),
    # read in param grid from config file
    # according to xgb doc, for early stopping having a list of eval metrics,
    # only the last one is used; therefore default 'logloss' is used hard coded
    param_grid = {
        'objective': config_file['model']['xgb_cls_cv']['objective'],
        'max_depth': config_file['model']['xgb_cls_cv']['max_depth'],
        'learning_rate': config_file['model']['xgb_cls_cv']['learning_rate'],
        'gamma': config_file['model']['xgb_cls_cv']['gamma'],
        'colsample_bytree': config_file['model']['xgb_cls_cv']['colsample_bytree'],
        # regarding regularisation ...
        'reg_alpha': config_file['model']['xgb_cls_cv']['reg_alpha'],
        'reg_lambda': config_file['model']['xgb_cls_cv']['reg_lambda'],
    }
    best_xgb_cls_cv = train_model(
        xgb.XGBClassifier(
            early_stopping_rounds=config_file['model']['early_stopping_rounds'],
            random_state=config_file['model']['random_seed']),
        X_train, y_train, X_test, y_test,
        param_grid=param_grid,
        config=config_file,
        test_run=None)
    logger.info("Best CV XGBClassifier model training params:\n %s", best_xgb_cls_cv.get_params())

    # retrain with best estimator params (call fit()),
    # without test data split; but such data is necessary for early stopping
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # https://towardsdatascience.com/how-to-do-cross-validation-effectively-1bbeb1d69ee8
    # https://stats.stackexchange.com/questions/586298/is-it-required-to-train-the-model-in-entire-data-after-cross-validation
    params = best_xgb_cls_cv.get_params()
    params['eval_metric'] = ['logloss']
    print('---')
    print(f'params dict of best cv xgb estimator with logloss eval_metric:\n{params}')
    print('---')

    best_xgb_cls = xgb.XGBClassifier(**params)
    best_xgb_cls.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1
    )
    logger.info("Retrained best CV XGBClassifier model fit params:\n %s", best_xgb_cls)

    # plot and store evaluation results diagram of retrained best xgb cv model    
    # plot logloss diagramm
    title = 'Best Retrained XGB CV Classifier: Resulting Loss Diagram'
    png_name = '_best_retrained_xgb-cv_logloss_treeNo_diagram.png'
    plot_logloss_diagram(best_xgb_cls, config_file, title, png_name)

    # plot classification error
    # Note: different eval metrics interfere with early stopping mechanism ...
    # title = 'Best XGB CV Classifier: Resulting Classification Error Diagram'
    # png_name = '_best_xgb-cv_class_error_diagram.png'
    # plot_error_diagram(best_xgb_cls, config_file, title, png_name)

    # plot ROC curve diagram
    title = 'Best Retrained XGB CV Classifier: Resulting ROC Curve Diagram'
    png_name = '_best_retrained_xgb-cv_roc-curve_diagram.png'
    plot_roc_curve_diagram(best_xgb_cls,
                           X_train, y_train,
                           X_test, y_test,
                           config_file, title, png_name)

    # prediction with test data;
    # regarding early stopping and getting optimal tree number, note:
    # with Scikit-Learn API of Xgboost param best_ntree_limit will be used directly
    y_preds_cv = inference(best_xgb_cls, X_test)

    # evaluation with test data and predictions
    precision, recall, fbeta, cm, cls_report = compute_model_metrics(y_test, y_preds_cv)
    print('Validation:')
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'fbeta : {fbeta}')
    print('Confusion Matrix')
    print(cm)
    print()
    print('Classification Report')
    print(cls_report)
    print('---------')
    logger.info("--- Validation metrics of Best CV XGBClassifier: ---")
    logger.info("Precision: %s, Recall: %s, Fbeta: %s", precision, recall, fbeta)
    logger.info('- Confusion Matrix -')
    logger.info(cm)
    logger.info('- Classification Report -')
    logger.info(cls_report)
    logger.info("----------")

    # save XGBoost cv model in model dir,
    # Note: during different runs: grid model has always been best compared to
    # single estimator without cv, therefore no check is implemented
    logger.info('Save retrained cross validation best XGBClassifier as final pickle file in models dir')
    best_xgb_label = ''.join([TODAY, '_', config_file['model']['xgb_cls_cv']['output_artifact']])
    filename = os.path.join(get_models_path(), best_xgb_label)
    with open(filename, 'wb') as f:
        joblib.dump(best_xgb_cls, f)

    # in general: having different model types check which one is the best would happen;
    # now we have only one, so, by now it is the same ... only with artifact label change
    artifact_label = config_file['model']['final_xgb_artifact']
    filename = os.path.join(get_models_path(), artifact_label)
    with open(filename, 'wb') as f:
        joblib.dump(best_xgb_cls, f)


if __name__ == "__main__":
    go()
