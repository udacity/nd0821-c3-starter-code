# import pytest

from fastapi.testclient import TestClient
from http import HTTPStatus
from main import app


client = TestClient(app)


def test_hello1():
    """
    Teste Get hello route
    """
    response = client.get('/')
    # assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"


def test_hello2():
    """
    Teste Get hello route
    """
    response = client.get('/')
    # assert response.status_code == HTTPStatus.OK
    assert response.json() == {'response_message': 'Hello', 'status_code': 200}


def test_hello3():
    """
    Teste Get hello route
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK


def test_predict_status1():
    """
    Tests POST route predict.
    """
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
    response = client.post("/model", json=data)
    assert response.request.method == "POST"


def test_predict_status2():
    """
    Tests POST route predict.
    """
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
    response = client.post("/model", json=data)
    assert response.status_code == HTTPStatus.OK


def test_predict_response1():
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 1,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 345234523,
            "capital-loss": 1,
            "hours-per-week": 50,
            "native-country": "United-States"}

    response = client.post("/model", json=data)
    assert response.json()["Result"] == 1


def test_predict_response0():
    data = {"age": 55,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Masters",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }

    response = client.post("/model", json=data)
    assert response.json()["Result"] == 0


def test_missing_feature_predict():
    """
    Testing post route for missing predict.
    """
    data = {
        "age": 0
    }
    response = client.post("/model", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
