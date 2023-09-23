#!/usr/bin/env -S python3 -i

"""
Script to handle slice performance evaluation of the XGBoost classifier,
only for unique value groups of categorical features of the preprocessed
dataset.
author: Ilona Brinkmeier
date:   2023-09

Note regarding preprocessing steps: 
sex is numerical, education_num used instead of education string
and the native-country feature is modified.

So, the categorical features are:
- "workclass",
- "marital-status",
- "occupation",
- "relationship",
- "race",
- "mod_native_country"
"""

###################
# Imports
###################
import os
import sys
import logging
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
TODAY = datetime.today().strftime('%Y-%m-%d_%H-%M')

# needed to run this script alone
MAIN_DIR = os.path.join(os.getcwd(), 'src/')
sys.path.append(MAIN_DIR)
sys.path.append(os.getcwd())
print(f'sys.path : {sys.path}')

from training.ml.model import inference, compute_model_metrics
from training.ml.data import clean_data, get_cat_features, process_data
from config import get_config, get_data_path, get_models_path

#####################
# Coding
#####################

# get logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)

# get configuration file
config_file = get_config()


def load_final_model_artifact():
    """ Returns final model classifier artifact from training process """
    path = os.path.join(get_models_path(), config_file['model']['final_xgb_artifact'])
    assert Path(path).exists(), f'model artifact of given path does not exist: {path}'
    logger.info('Final model artifact path for slice performance exists: %s', path)
    try: 
        return joblib.load(path)
    except Exception as e:
        logger.exception("Census final model artifact not loaded for slice performance action")
        raise FileNotFoundError(path) from e


def load_transformer_artifact():
    """ Returns final column transformer artifact from training process """
    path = os.path.join(get_models_path(), config_file['model']['col_transformer'])
    assert Path(path).exists(), f'transformer artifact of given path does not exist: {path}'
    logger.info('Final column transformer artifact path for slice performance exists: %s', path)
    try: 
        return joblib.load(path)
    except Exception as e:
        logger.exception("Census final transformer artifact not loaded for slice performance action")
        raise FileNotFoundError(path) from e


def load_preproc_data() -> pd.DataFrame:
    """ 
    Returns the preprocessed census dataset as dataframe,
    if available. If not raises FileNotFoundError exception.
    """
    path = os.path.join(get_data_path(), config_file['etl']['orig_census'])
    assert Path(path).exists(), f'dataset of given path does not exist: {path}'
    logger.info('Census preprocessed data path for slice performance exists: %s', path)
    try:
        df = pd.read_csv(path)
        df = clean_data(df, config_file)
        logger.info('Census preprocessed dataset: %s samples with %s features each', \
                    df.shape[0], df.shape[1])
        logger.info('Its columns are: %s', df.columns)
        return df
    except Exception as e:
        logger.exception("Census preprocessed data not loaded for slice performance action")
        raise FileNotFoundError(path) from e


def compute_slice_metrics_categorical():
    """
    Calculates metrics on data slices of specific categorical columns via inference.
    The metrics are Precision, Recall, F1 and Confusion Matrix.
    Evaluation information is stored in output file slice_output.txt.
    
    Note:
    If as starting point data or pretrained artifacts could not be read in, system exit happens.
    
    Using the model created before on the same data, means this is only a playground
    of a testing procedure and not a valid production one. Using this for production,
    new, unknown data must be used.
    The metric results are not expected to be valid, because the model knows data already
    and data leakage issue may appear, overfitting is expected.
    """
    
    # read in cleaned data, remember: salary is a binary number after cleaning
    try:
        df_data = load_preproc_data()
        _, test = train_test_split(df_data, test_size=0.20)
        cat_features = get_cat_features(df_data)
        model_artifact = load_final_model_artifact()
        transformer_artifact = load_transformer_artifact()
    except Exception as e:
        logger.exception("Exit: exception of type %s occurred. Details: %s", type(e).__name__, str(e))
        sys.exit(1)
    
    # write to performance metric file - './<date>_slice_output.txt'
    FILE_HEADER = 'Performance Evaluation Metrics of Categorical Census Features\n'
    LINE = '_____________________________________________________________\n\n'
    file_name = os.path.join(MAIN_DIR, TODAY + '_' + config_file['model']['slice_output_file'])
    
    slice_metrics = []
    for cat_feat in cat_features:
        for cols in test[cat_feat].unique(): # look cols of categorical unique group
            
            logger.info('Slice performance evaluation for categorical value group and cols: %s',\
                        cols)
            # prepare data
            df_cat_group = test[test[cat_feat] == cols]
            y = df_cat_group['salary']
            X = df_cat_group.drop(columns=['salary'])
            X_processed = transformer_artifact.transform(X) # inference case, not training
            
            # predict
            y_preds = inference(model_artifact, X_processed)              
            
            # metrics, cls-report not used
            prec, rec, fbeta, cm, _ = compute_model_metrics(
                y, y_preds
            )
            cat_group_metrics = (
                f'{cat_feat} - {cols}:\n'
                f'   Precision: {prec: .2f}, Recall: {rec: .2f}, Fbeta: {fbeta: .2f}\n'
                f'   Confusion Matrix: \n{cm}\n\n'
            )
            slice_metrics.append(cat_group_metrics)

            # write to performance metric file - './<date>_slice_output.txt'
            with open(file_name, 'w') as a_writer:
                a_writer.write(FILE_HEADER)
                a_writer.write(LINE)
                for cat_group_metrics in slice_metrics:
                    a_writer.write(cat_group_metrics)

    logging.info("Performance evaluation metrics for categorical slices saved to slice output text file")


if __name__ == '__main__':
    compute_slice_metrics_categorical()
