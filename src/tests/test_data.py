"""
Testsuite for data checks
author: I. Brinkmeier
date:   2023-08
"""

# 
# Deterministic Tests
#

def test_orig_not_empty(data):
    """ Checks that the original dataset is not empty. """
    df = data
    assert len(df) > 0
    

def test_orig_duplicated_rows(data):
    """ Checks that duplicate rows exists in original dataset. """
    df = data
    duplicated = df.duplicated().sum()
    assert duplicated > 0