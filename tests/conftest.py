#!/usr/bin/env -S python3 -i

"""
This conftest script delivers the fixture setup used for pytest testsuite.
author: I. Brinkmeier
date:   2023-09

for information about testing see e.g.:
- https://docs.pytest.org/en/7.1.x/how-to/fixtures.html
- https://www.testim.io/blog/using-pytest-fixtures/
- https://www.testim.io/blog/python-test-automation/
"""

###################
# Imports
###################
import os
import sys
import pandas as pd
import pytest

from src.config import get_config, get_data_path
from src.training.ml.data import clean_data

###################
# Coding
###################

#
# arrange setup
#
TEST_DIR = os.path.join(os.getcwd(), 'tests/')
sys.path.append(TEST_DIR)

# load in the configuration file
config_file = get_config()


@pytest.fixture(scope='session', autouse=True)
def raw_test_data() -> pd.DataFrame:
    """
    Original dataframe loaded from csv file. First 1500 rows are used for testing.

    Returns:
        Subset dataframe with 1500 rows loaded from original csv file
    """
    data = os.path.join(get_data_path(), config_file['etl']['orig_census'])

    if not os.path.exists(data):
        pytest.fail(f"Fixture creation with 1500 orig rows: Data not found at path: {data}")

    return pd.read_csv(data)[:1500]


@pytest.fixture(scope='session')
def cleaned_test_data() -> pd.DataFrame:
    """
    First 200 rows of cleaned, preprocessed raw test data.

    Returns:
        Subset dataframe with 200 rows loaded from original csv file and cleaned
    """
    data = os.path.join(get_data_path(), config_file['etl']['orig_census'])

    if not os.path.exists(data):
        pytest.fail(f"Fixture creation with 200 cleaned rows: Data not found at path: {data}")

    return clean_data(pd.read_csv(data)[:200], config_file)
