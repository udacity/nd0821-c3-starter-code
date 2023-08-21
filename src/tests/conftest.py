"""
Testsuite for data checks
author: I. Brinkmeier
date:   2023-08

for information about testing see e.g.: 
- https://www.testim.io/blog/using-pytest-fixtures/
- https://www.testim.io/blog/python-test-automation/
"""

###################
#
# Imports
#
###################
import os
import pytest
import pandas as pd
import yaml

###################
#
# Coding
#
###################

# read config file
CONFIG_FILE = '../../config.yml'
with open(CONFIG_FILE, 'r') as f:
    try:
        config = yaml.safe_load(f.read())
    except yaml.YAMLError as exc:
        print(f'Cannot read config file: {exc}')


@pytest.fixture(scope='session')
def data():
    """
    Original dataset loaded from csv file used for testing

    Returns:
        df : Dataframe with original data loaded from csv file
    """
    
    if not os.path.exists(str(config['etl']['orig_census'])):
        pytest.fail(f"Data not found at path: {str(config['etl']['orig_census'])}")
    df = pd.read_csv(str(config['etl']['orig_census']))

    return df
