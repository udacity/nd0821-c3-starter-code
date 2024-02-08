#!/usr/bin/env -S python3 -i

"""
This conftest script delivers the fixture setup used for pytest testsuite.
Having had issues with dvc cache and github actions part of unittests,
a subset of data is stored once in the tests directory.

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

from src.config import get_config
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
    try:
        ROOT = os.getcwd()
        filepath = os.path.join(ROOT, config_file['etl']['test_data_orig_census'])
        return pd.read_csv(filepath)
    except Exception as e:
        pytest.fail(f"Fixture with 1500 orig rows: exc: {e}")


@pytest.fixture(scope='session')
def cleaned_test_data() -> pd.DataFrame:
    """
    First 200 rows of cleaned, preprocessed raw test data.

    Returns:
        Subset dataframe with 200 rows loaded from original csv file and cleaned
    """
    try:
        ROOT = os.getcwd()
        filepath = os.path.join(ROOT, config_file['etl']['test_data_orig_census'])
        return clean_data(pd.read_csv(filepath)[:200], config_file)
    except Exception as e:
        pytest.fail(f"Fixture with 200 cleaned rows: exc: {e}")
