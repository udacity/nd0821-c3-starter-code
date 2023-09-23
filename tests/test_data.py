#!/usr/bin/env -S python3 -i

"""
Testsuite for data checks.
author: I. Brinkmeier
date:   2023-09
"""

###################
# Imports
###################
import pandas as pd
import pytest

from typing import Callable
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.config import get_config
from src.training.ml.data import process_data

###################
# Coding
###################

#
# arrange setup
#
# load in the configuration file
config_file = get_config()


# 
# Deterministic Tests
#

def test_not_empty(raw_test_data: Callable) -> None:
    """ Checks that original test dataset (dataframe instance) is not empty. """
    assert isinstance(raw_test_data, pd.DataFrame)
    assert len(raw_test_data) > 0, 'Dataframe is empty'
    assert raw_test_data.shape[0]>0, 'Dataframe has no rows'
    assert raw_test_data.shape[1]>0, 'Dataframe has no columns'
    

def test_duplicated_rows(cleaned_test_data: Callable) -> None:
    """ Checks if duplicate rows exists in test dataset after preprocessing. """
    assert cleaned_test_data.duplicated().sum() == 0, 'Dataframe includes duplicates'
    

def test_columns_exist(cleaned_test_data: Callable) -> None:
    """ Checks if the expected column labels are available after preprocessing
        as part of the expected cleaning process.
    """    
    preproc_columns = ['age', 'workclass', 'education_num', 'marital_status',
                        'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                        'capital_loss', 'hours_per_week', 'salary', 'mod_native_country']
    
    assert cleaned_test_data.columns.to_list() == preproc_columns, \
        'For column labels its feature preproc not as expected'

    
def test_data_types(cleaned_test_data) -> None:
    """ Checks if the expected column datatypes are available after preprocessing
        as part of the expected cleaning process.
    """
    data_types = {
        'age': 'int64',
        'workclass': 'object',
        'education_num': 'int64',
        'marital_status': 'object',
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'int64',
        'capital_gain': 'int64',
        'capital_loss': 'int64',
        'hours_per_week': 'int64',
        'mod_native_country': 'object',
        'salary': 'int64',
    }
    
    assert cleaned_test_data.dtypes.map(str).to_dict() == data_types, \
        'Data types are not as expected for 13 preproc columns'
    
    
def test_process_data_with_scaling(raw_test_data: Callable) -> None:
    """ Checks return of ColumnTransformer instance and
        X and y as pipeline output with scaling=True parameter.
    """
    processor, X, y = process_data(raw_test_data,
                                   label='salary',
                                   scaling=True,
                                   config_file=config_file)
    assert isinstance(processor, ColumnTransformer), 'Process data does not return ColumnTransformer'
    assert len(processor.transformers) == 3, 'Different transformer steps than 3'
    assert len(X) > 0, 'X dataframe is empty'
    assert len(y) > 0,  'y target column is empty'
    
    # 'num_transf' includes difference regarding scaling
    assert isinstance(processor.transformers[0][1], Pipeline), 'First step, numerical: Pipeline not given'
    assert len(processor.transformers[0][1]) == 2, 'Numerical pipeline includes different step number than 2'
    assert isinstance(processor.transformers[0][1][0], SimpleImputer), 'numerical pipe: StandardScaler missing' 
    assert isinstance(processor.transformers[0][1][1], StandardScaler), 'with scaling: StandardScaler missing'  


if __name__ == "__main__":
    # as cli command to create the data test report: 
    # pytest ./tests/test_data.py --html=./tests/data_test_report.html --capture=tee-sys
    pytest.main(["--html=data_test_report.html", "-v"])