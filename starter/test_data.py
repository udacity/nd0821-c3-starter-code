'''
clean data testing

Author: Oliver
Date: February 2022
'''
import pandas as pd


def test_column_presence_and_type(data):
    '''
    check for matching columns & types
    '''
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,  
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype
    }

    # Check column presence  
    assert set(data.columns).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_check_duplicates(data):
    '''
    check for duplicate rows
    '''
    assert(data.duplicated().any() == False)
