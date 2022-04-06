import json
import pytest
import sys

from fastapi.testclient import TestClient

# module error
try:
    from main import app
except ModuleNotFoundError:
    sys.path.append('./')
    from main import app

@pytest.fixture(scope="session")
def client():
    client = TestClient(app)
    return client

def test_get(client):
    """Test Get route"""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {'message': 'Hello'}

def test_incorrect_path(client):
    """Path incorrect or not found"""

    res = client.get("/some_nonexistent_url")

    assert res.status_code != 200
    assert res.json() == {"detail":"Not Found"}


def test_post(client):
    res = client.post("/model", json={
                    "age": 55,
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
                    "native-country": "United-States",
            })

    assert res.status_code == 200
    assert res.json() == {"Result": 0}

def test_post_method(client):
    """Test the post method

    Args:
        client : request
    """    
    res = client.post("/model", json={
                    "age": 55,
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
            })

    assert res.status_code != 200
    assert json.loads(res.content)['detail'][0]['type'] == "value_error.missing"

