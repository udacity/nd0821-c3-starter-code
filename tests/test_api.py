#!/usr/bin/env -S python3 -i

"""
Testsuite for FastAPI checks.
For response attributes and methods see:
https://www.geeksforgeeks.org/response-methods-python-requests/


author: I. Brinkmeier
date:   2023-09
"""

###################
# Imports
###################
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from src.main import app


###################
# Coding
###################


def test_root():
    """
    Tests GET root function for greeting
    """
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        assert response.content == b"Welcome to the Udacity MLOps project 3 and its salary prediction application!"


@pytest.mark.parametrize('test_input, expected',
                         [
                            ('age', "Person's age - numerical value (int)"),
                            ('marital_status', "Person's marital status - nominal categorical value (str)")
                         ]
)
def test_feature_info_response(test_input: str, expected: str):
    """
    Tests GET request response of function feature_labels()

    Args:
        test_input (str): example input
        expected (str): example output
    """
    with TestClient(app) as client:
        response = client.get(f'/feature_labels/{test_input}')
        assert response.json() == expected


def test_predict_status():
    """
    Tests POST predict function status with first sample of dataset
    """
    sample = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }

    with TestClient(app) as client:
        response = client.post("/predict/", json=sample)
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "POST"


def test_missing_feature_predict():
    """
    Tests POST predict function about failure due to missing features.
    Regarding the associated example for FastAPI, it starts with missing age feature.
    """
    data = {}
    with TestClient(app) as client:
        response = client.post("/predict/", json=data)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY   # no. 422
        assert response.request.method == "POST"
        assert response.json()["detail"][0]["type"] == "value_error.missing"


def test_predict_response_negative():
    """
    Tests POST predict function about successful response with first sample of dataset.
    'salary' of first sample is '<=50K'.
    """
    sample = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=sample)
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "POST"
        assert response.content == b'income prediction label: 0, salary class: <=50k'
        # using CLI pytest command with python 3.10.9 the following line throws
        # json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
        # but results in a successful sanitycheck, this Udacity script
        # checks response.json only, but not a FastAPI Response instance
        # assert response.json()['salary class'] == '<=50k'


def test_predict_response_positive():
    """
    Tests POST predict function about successful response with sample no 8 of dataset.
    'salary' of this sample is '>50K'.
    """
    sample = {
        'age': 31,
        'workclass': 'Private',
        'fnlgt': 45781,
        'education': 'Masters',
        'education_num': 14,
        'marital_status': 'Never-married',
        'occupation': 'Prof-specialty',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital_gain': 14084,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=sample)
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "POST"
        assert response.content == b'income prediction label: 1, salary class: >50k'
        # using CLI pytest command with python 3.10.9 the following line throws
        # json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
        # but results in a successful sanitycheck, this Udacity script
        # checks response.json only, but not a FastAPI Response instance
        # assert response.json()['salary class'] == '>50k'


def test_wrong_feature_type_str_predict():
    """
    Tests POST predict function about failure due to wrong feature types.
    Regarding the associated example, it starts with education_num feature
    followed by hours_per_week both delivered as string.
    How is the ability of the model to get an inference value?
    It is expected that this few changes can be handled, because FastAPI
    and json are working with strings. So, a proper transformation with
    HTTPStatus.OK (no. 200) shall be the test result.
    """
    sample = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': '13',
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': '40',
        'native_country': 'United-States'
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=sample)
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "POST"
        assert response.content == b'income prediction label: 1, salary class: >50k' or \
               response.content == b'income prediction label: 0, salary class: <=50k'


def test_wrong_feature_type_int_predict():
    """
    Tests POST predict function about failure due to wrong feature types.
    Regarding the associated example, it starts with workclass feature
    followed by relationship both delivered as int.
    How is the ability of the model to get an inference value?
    It is expected that this few changes can be handled, because FastAPI
    and json transform the unknown categories to zero or None. So, a proper
    transformation with HTTPStatus.OK (no. 200) shall be the test result.
    """
    sample = {
        'age': 39,
        'workclass': 3,
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 12,
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=sample)
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "POST"
        assert response.content == b'income prediction label: 1, salary class: >50k' or \
               response.content == b'income prediction label: 0, salary class: <=50k'


if __name__ == "__main__":
    # as cli command to create the api test report:
    # pytest ./tests/test_api.py --html=./tests/api_test_report.html --capture=tee-sys
    pytest.main(["--html=api_test_report.html", "-v"])
