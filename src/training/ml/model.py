#!/usr/bin/env -S python3 -i

###################
# Imports
###################
import os
import logging

from datetime import datetime
TODAY = datetime.today().strftime('%Y-%m-%d_%H-%M')

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)
from xgboost import XGBClassifier

###################
# Coding
###################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)


def plot_error_diagram(estimator, config, title, png_name):
    ''' Stores classification error evaluation results diagram '''
    results = estimator.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(10,7))
    plt.title(title, fontsize=13, fontweight='bold')
    ax.plot(x_axis, results["validation_0"]["error"], label="Training")
    ax.plot(x_axis, results["validation_1"]["error"], label="Testing")
    plt.axvline(estimator.best_ntree_limit,
                color="gray", label="Optimal tree number")
    plt.xlabel("Tree Numbers")
    plt.ylabel("Classification Error")
    plt.legend()
    plot_name = ''.join([TODAY, png_name])
    plt.savefig(os.path.join(os.getcwd(), config['eda']['plots_path'], str(plot_name)),
                bbox_inches='tight')


def plot_logloss_diagram(estimator, config, title, png_name):
    ''' Stores classification logloss evaluation results diagram '''
    results = estimator.evals_result()
    epochs = len(results["validation_0"]["logloss"])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(10,7))
    plt.title(title, fontsize=13, fontweight='bold')
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Training")
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Testing")
    plt.axvline(estimator.best_ntree_limit,
                color="gray", label="Optimal tree number")
    plt.xlabel("Tree Numbers")
    plt.ylabel("Loss")
    plt.legend()
    plot_name = ''.join([TODAY, png_name])
    plt.savefig(os.path.join(os.getcwd(), config['eda']['plots_path'], str(plot_name)),
                bbox_inches='tight')


def plot_roc_curve_diagram(estimator, X_train, y_train, X_test, y_test, config, title, png_name):
        ''' Stores plotted ROC curve diagram in plots directory '''
        fig, ax = plt.subplots(figsize=(10,7))
        plt.title(title, fontsize=13, fontweight='bold')
        RocCurveDisplay.from_estimator(estimator,
                                       X_train, y_train, ax=ax, label='Training')
        RocCurveDisplay.from_estimator(estimator,
                                       X_test, y_test, ax=ax) #, label='Testing')
        plt.legend()
        plot_name = ''.join([TODAY, png_name])
        plt.savefig(os.path.join(os.getcwd(), config['eda']['plots_path'], str(plot_name)),
                    bbox_inches='tight')
    

def train_model(model, X_train, y_train, X_test, y_test, param_grid=None, config=None, test_run=None):
    """
    Returns a trained machine learning model with or without hyperparameter tuning
    from GridSearchCV().
    Creates the evaluation set from given parameters: [(X_train, y_train), (X_test, y_test)]
    used for early stopping.

    Inputs
    ------
    model: 
        machine learning model we have to train during training process,
        expected is XGBoost classifier
    X_train : pd.dataframe
        Training data features.
    y_train : pd.dataframe
        Training target labels.
    X_test : pd.dataframe
        Testing data features.
    y_test : pd.dataframe
        Testing target labels.
    param_grid : dictionary from config.py
        Hyperparameters for CrossValidation
    config: dictionary
        To get model configuration settings
    test_run: Boolean or None
        By default None, for application usage,
        is set to True for test suite run
        
    Returns
    -------
    model : scikit-learn model or pipeline
        Trained machine learning model.
    """
         
    # Future toDo:
    # - behavioral strategy pattern for different classifiers
    
    try:
        assert isinstance(model, (XGBClassifier))
    except AssertionError as error:
        logging.error("Model should be XGBClassifier %s", error)
    
    if param_grid: # cross validation process
        if config:
            n_jobs = config['model']['n_jobs']
            cv = config['model']['cv']
            verbose = config['model']['verbose']
        else:
            n_jobs = -1
            cv = 5
            verbose = 1
            
        grid_cv = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               n_jobs=n_jobs,
                               cv=cv,
                               verbose=verbose)
        
        # usage of early stopping
        grid_result = grid_cv.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)])
        
        best_estimator = grid_cv.best_estimator_
                
        if test_run:
            return best_estimator
        
        # Print the best score and corresponding hyperparameters
        print('---- Prediction Result: GridSearchCV - XGBClassifier ---')
        print(f'Best score is {grid_result.best_score_:.4f}')
        print(f'Best hyperparameters are {grid_result.best_params_}')
        print(f'Optimal number of trees is {best_estimator.best_ntree_limit}')
        print(f'Best cv estimator: {best_estimator}')
        print('----')
        logger.info('---- Prediction Result: GridSearchCV - XGBClassifier ---')
        logger.info('Best score is %s', grid_result.best_score_)
        logger.info('Best hyperparameters are %s', grid_result.best_params_)
        logger.info('Optimal number of trees is %s', best_estimator.best_ntree_limit) 
        logger.info('Best cv estimator: %s', best_estimator)
        logger.info('----')
        
        return best_estimator
    else: # single model process
        # usage of early stopping
        model = model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)])
                
        if test_run:
            return model
        
        print('---- Prediction Result: Single basic - XGBClassifier model ---')
        print(model)
        print(f'Optimal number of trees is {model.best_ntree_limit}')
        logger.info('---- Prediction Result: Single basic - XGBClassifier model ---')
        logger.info(model)
        logger.info('Optimal number of trees is %s', model.best_ntree_limit)
        logger.info('----')

        return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, F1, 
    confusion matrix and whole classification report.

    Inputs
    ------
    y : np.array
        Known labels, binarized. (refers to y_test)
    preds : np.array
        Predicted labels, binarized. (refers to y_preds)
        
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    cm : array  (confusion matrix)
    cls_report: whole classification report as dictionary
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    cm = confusion_matrix(y, preds)
    cls_report = classification_report(y, preds, output_dict=True)
    
    return precision, recall, fbeta, cm, cls_report


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : scikit-learn classification model or pipeline
        Trained machine learning model.
    X : pd.DataFrame
        Test data used for prediction.
        
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
