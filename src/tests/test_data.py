#!/usr/bin/env -S python3 -i

"""
Testsuite for data checks
author: I. Brinkmeier
date:   2023-08
"""

###################
# Imports
###################
import logging

###################
# Coding
###################

# set logging properties
# info see: https://realpython.com/python-logging-source-code/
logger = logging.getLogger(__name__)

# 
# Deterministic Tests
#

def test_orig_not_empty(data):
    """ Checks that the original dataset is not empty. """
    df = data
    assert len(df) > 0
    

def test_orig_duplicated_rows(data):
    """ Checks if duplicate rows exists in original dataset. """
    df = data
    duplicated = df.duplicated().sum()
    assert duplicated > 0